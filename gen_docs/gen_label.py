import sys
import json


if __name__ == "__main__":
    doc_type = sys.argv[1]
    path = sys.argv[2]
    data = {}

    data["type"] = doc_type
    if doc_type=='api':
        import megengine
        data["version"] = megengine.__version__
    
    with open(path,"w") as f:
        f.write(json.dumps(data))

