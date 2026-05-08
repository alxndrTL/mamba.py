// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "MetalMamba",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(
            name: "MetalMamba",
            type: .dynamic,
            targets: ["MetalMamba"]
        ),
        .library(
            name: "MetalMambaBridge",
            type: .dynamic,
            targets: ["MetalMambaBridge"]
        ),
        .executable(
            name: "test-pscan",
            targets: ["TestPScan"]
        )
    ],
    targets: [
        .target(
            name: "MetalMamba",
            path: "Sources/MetalMamba",
            resources: [
                .process("pscan.metal")
            ]
        ),
        .target(
            name: "MetalMambaBridge",
            dependencies: ["MetalMamba"],
            path: "Sources/MetalMambaBridge"
        ),
        .executableTarget(
            name: "TestPScan",
            dependencies: ["MetalMamba"],
            path: "Sources/TestPScan"
        )
    ]
)
