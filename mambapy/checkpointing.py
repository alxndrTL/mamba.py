import torch

"""

Activation checkpointing for stateful sequence models (Mamba, etc).

Standard PyTorch checkpointing (torch.utils.checkpoint) doesn't work with pscan
because it operates in-place. This module provides a custom checkpointing mechanism
that stores only the small state tensors at segment boundaries, not the full activations.

The idea is to split a long sequence into segments and process them one at a time :
- Forward : run all segments without saving activations, store only states at segment boundaries
- Backward : recompute activations one segment at a time, propagating gradients through states

This gives us BPTT (backpropagation through time) with constant memory usage.

Memory : O(1 segment's activations + n_segments * state_size) instead of O(n_segments * activations)
Compute : 2x forward (once in forward, once in backward for recomputation)

"""

class SequenceCheckpoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dummy, model, n_layers, tuple_size, num_state_tensors, *tensors):
        # dummy : (1,) requires_grad=True, anchor for autograd graph
        # model : callable (x, state) -> (y, new_state)
        # n_layers : number of layers
        # tuple_size : size of each layer's state tuple
        # num_state_tensors : n_layers * tuple_size
        # *tensors : (*initial_state_flat, *segments)

        ctx.model = model
        ctx.n_layers = n_layers
        ctx.tuple_size = tuple_size
        ctx.num_state_tensors = num_state_tensors

        initial_state_flat = tensors[:num_state_tensors]
        segments = tensors[num_state_tensors:]

        ctx.initial_state = tuple(t.detach() if t is not None else None for t in initial_state_flat)
        ctx.state_requires_grad = [t.requires_grad if t is not None else False for t in initial_state_flat]
        ctx.num_segments = len(segments)

        state_flat = list(initial_state_flat)
        outputs = []
        states_at_boundaries = [] # states at the beginning of each segment

        # forward without saving activations
        with torch.no_grad():
            for x in segments:
                states_at_boundaries.append(
                    [t.detach().clone() if t is not None else None for t in state_flat]
                )

                state = [
                    tuple(t.detach() if t is not None else None
                          for t in state_flat[j * tuple_size : (j + 1) * tuple_size])
                    for j in range(n_layers)
                ]
                y, new_state = model(x.detach(), state)

                outputs.append(y.detach().requires_grad_(True))
                state_flat = [
                    t.clone() if t is not None else None
                    for layer_tuple in new_state for t in layer_tuple
                ]

        final_state = tuple(t.requires_grad_(True) if t is not None else None for t in state_flat)

        # save segments and boundary states for backward recomputation
        states_flat_for_saving = [t for sl in states_at_boundaries for t in sl if t is not None]
        ctx.states_none_mask = [[t is None for t in sl] for sl in states_at_boundaries]
        ctx.save_for_backward(*segments, *states_flat_for_saving)

        return (*outputs, *final_state)

    @staticmethod
    def backward(ctx, *grad_outputs):
        num_segments = ctx.num_segments
        grad_ys = list(grad_outputs[:num_segments])
        grad_final_state = list(grad_outputs[num_segments:])

        all_saved = ctx.saved_tensors
        segments = all_saved[:num_segments]
        saved_states_flat = all_saved[num_segments:]

        # reconstruct boundary states
        states = {}
        it = iter(saved_states_flat)
        for i, none_mask in enumerate(ctx.states_none_mask):
            states[i] = [None if is_none else next(it) for is_none in none_mask]

        # process segments in reverse, propagating gradients through states
        current_grad_state = grad_final_state
        grad_segments = {}

        for i in range(num_segments - 1, -1, -1):
            state_at_i = [t.detach().requires_grad_(True) if t is not None else None for t in states.pop(i)]

            seg_detached = segments[i].detach().requires_grad_(True)
            with torch.enable_grad():
                state = [
                    tuple(state_at_i[k * ctx.tuple_size : (k + 1) * ctx.tuple_size])
                    for k in range(ctx.n_layers)
                ]
                y, new_state = ctx.model(seg_detached, state)
                flat_new_state = [t for layer_tuple in new_state for t in layer_tuple]

            # combine output grad and state grad from later segments
            outputs = [y] + flat_new_state
            grads = [grad_ys[i]] + current_grad_state

            filtered = [(o, g) for o, g in zip(outputs, grads) if g is not None]
            if not filtered:
                current_grad_state = [None] * ctx.num_state_tensors
                grad_segments[i] = None
                continue

            filtered_outputs, filtered_grads = zip(*filtered)

            state_is_not_none = [t is not None for t in state_at_i]
            state_inputs = [t for t in state_at_i if t is not None]

            grad_targets = state_inputs + [seg_detached]
            if grad_targets:
                all_grads = list(torch.autograd.grad(
                    filtered_outputs, grad_targets, filtered_grads,
                    retain_graph=True, allow_unused=True,
                ))

                num_si = len(state_inputs)
                state_grads = all_grads[:num_si]
                grad_segments[i] = all_grads[num_si] if num_si < len(all_grads) else None

                if state_inputs:
                    current_grad_state = []
                    gi = iter(state_grads)
                    for not_none in state_is_not_none:
                        current_grad_state.append(next(gi) if not_none else None)
                else:
                    current_grad_state = [None] * ctx.num_state_tensors
            else:
                current_grad_state = [None] * ctx.num_state_tensors
                grad_segments[i] = None

            # backward through model parameters
            torch.autograd.backward(filtered_outputs, filtered_grads, retain_graph=False)

        # only return gradients for initial state positions that had requires_grad
        full_grad_initial_state = [
            current_grad_state[j] if ctx.state_requires_grad[j] else None
            for j in range(ctx.num_state_tensors)
        ]
        seg_grads = [grad_segments.get(i) for i in range(num_segments)]

        return (None, None, None, None, None, *full_grad_initial_state, *seg_grads)


def checkpointed_sequence(model, segments, initial_state):
    # model : callable (x, state) -> (y, new_state), eg model.chunk_step
    # segments : list of (B, seg_len, D)
    # initial_state : [(h, inputs), ...] per layer, h can be None
    #
    # returns (outputs, final_state)

    if not segments:
        return [], list(initial_state)

    n_layers = len(initial_state)
    tuple_size = len(initial_state[0]) if initial_state else 0
    initial_state_flat = [t for layer_tuple in initial_state for t in layer_tuple]

    # dummy tensor to anchor the autograd graph (needed when inputs don't require grad)
    device = segments[0].device if segments else 'cpu'
    dtype = segments[0].dtype if segments else torch.float32
    dummy = torch.zeros(1, device=device, dtype=dtype, requires_grad=True)

    results = SequenceCheckpoint.apply(
        dummy, model, n_layers, tuple_size, len(initial_state_flat),
        *initial_state_flat, *segments,
    )

    num_segments = len(segments)
    outputs = list(results[:num_segments])
    final_state_flat = results[num_segments:]
    final_state = [
        tuple(final_state_flat[i * tuple_size : (i + 1) * tuple_size])
        for i in range(n_layers)
    ]

    return outputs, final_state
