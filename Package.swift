// swift-tools-version: 5.10
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "Whisper",
	platforms: [
		.iOS(.v16),
		.macOS(.v13),
		.macCatalyst(.v13)
	],
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
        .library(
            name: "Whisper",
            targets: ["Whisper"]),
    ],
	dependencies: [
		.package(url: "https://github.com/argmaxinc/WhisperKit", branch: "main")
	],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        .target(
			name: "Whisper", dependencies: [
				.product(name: "WhisperKit", package: "whisperkit")
			]
		),
        .testTarget(
            name: "WhisperTests",
            dependencies: ["Whisper"]),
    ]
)
