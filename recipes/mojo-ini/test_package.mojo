from ini import parse, to_ini

fn main() raises:
    # Parse a simple INI string
    var cfg = parse("""
[App]
name = mojo-ini
version = 0.2.0
""")
    if cfg["App"]["name"] != "mojo-ini":
        raise Error("Parse failed: expected name=mojo-ini")

    # Write a simple INI
    var data = Dict[String, Dict[String, String]]()
    data["App"] = Dict[String, String]()
    data["App"]["name"] = "mojo-ini"
    var out = to_ini(data)
    if not ("[App]" in out and "name = mojo-ini" in out):
        raise Error("Write failed: expected [App] with name = mojo-ini")

    print("ok")
