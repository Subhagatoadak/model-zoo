import argparse
import json
from model_zoo.registry.models import ModelRegistry
from model_zoo.registry.metrics import MetricRegistry

def main():
    parser = argparse.ArgumentParser(prog="mzoo", description="Model Zoo CLI")
    parser.add_argument("cmd", choices=["list-models", "list-metrics"])
    args = parser.parse_args()

    if args.cmd == "list-models":
        reg = ModelRegistry()
        print(json.dumps([m.to_dict() for m in reg.list()], indent=2))
    elif args.cmd == "list-metrics":
        reg = MetricRegistry()
        print(json.dumps([m.to_dict() for m in reg.list()], indent=2))

if __name__ == "__main__":
    main()
