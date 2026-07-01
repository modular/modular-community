from clang.cindex import Index


def main() raises:
    var index = Index.create()
    _ = index
    print("libclang_mojo package smoke test passed")
