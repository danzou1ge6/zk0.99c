# Accelerating zk-SNARK with GPU

## Composition
- `mont` holds implementation for Montegomery field.
- `doc` holds development documentation

## Build

To build everything,
```sh
xmake
```

To test implementation for Montgomery field,
```sh
xmake run test-mont
```

## Dev Documentation

Composed with [mdBook](https://github.com/rust-lang/mdBook).
To read it,
```sh
cd doc
mdbook build --open
```
