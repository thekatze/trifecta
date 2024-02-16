# trifecta

my third try at learning low level game development.
powered by wgpu

## running

### desktop
```sh
$ cargo run
```

### browser
this project utilizes wasm-server-runner for convenience, install it first

```sh
$ cargo install wasm-server-runner
```

then cargo run specifying the wasm target and open the url it prints in a webgpu-compatible browser

```sh
$ cargo run --target wasm32-unknown-unknown
```

