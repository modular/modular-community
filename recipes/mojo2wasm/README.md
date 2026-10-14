# Mojo2Wasm

An experimental, self-hosted compiler frontend written **entirely in Mojo** to compile Mojo source code into WebAssembly (Wasm). 

The goal of this project is to bridge the gap between Mojo's high-performance systems programming capabilities and the web/edge ecosystem.

## Why Mojo2Wasm?

While official WebAssembly support is still maturing in the core toolchain, `mojo2wasm` takes a grassroots, native approach. By building the compiler in Mojo itself, this project dogfoods the language's strict memory management, powerful `struct` system, and performance capabilities to generate optimized Wasm binaries.

## How It Works (Architecture)

The compiler bypasses heavy toolchain dependencies by implementing a custom compilation pipeline:

1. **Lexing & Parsing:** Scans native Mojo source code and converts it into a structured Abstract Syntax Tree (AST) using highly efficient Mojo data structures.
2. **Type Checking & Lowering:** (In Progress) Maps Mojo's strict type system and object model down to WebAssembly primitives.
3. **Code Generation:** Translates the valid AST nodes directly into WebAssembly format, utilizing Wasm's linear memory model.

## 📁 Repository Structure

* `/src`: Core compiler source code (Lexer, Parser, AST structures, and Code Generator).
* `/examples`: Sample Mojo files intended to be compiled to Wasm.
* `/tests`: Test suites verifying AST generation and Wasm output compliance.
* `main.mojo`: The primary entry point for the CLI tool.

## 🚧 Current Project Status

This project is a **Work in Progress (WIP)**. 

### Roadmap:
- [x] Lexer & Tokenizer for basic Mojo syntax
- [ ] AST Node definitions for variables, functions, and control flow
- [ ] WebAssembly Text Format (`.wat`) emitter
- [ ] WebAssembly Binary Format (`.wasm`) encoder
- [ ] Memory allocator mapping (Mojo pointers to Wasm linear memory)
- [ ] Support for Mojo `fn` and `struct` definitions

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.