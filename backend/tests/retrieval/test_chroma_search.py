from app.retrieval.chroma_search import ChromaVectorSearch


class Collection:
    def __init__(self):
        self.query_args = None

    def query(self, **kwargs):
        self.query_args = kwargs
        return {
            "documents": [["code"]],
            "metadatas": [[{"file_path": "src/a.py", "start_line": 2, "end_line": 4}]],
            "distances": [[0.2]],
        }


class Client:
    def __init__(self):
        self.collection = Collection()
        self.names = []

    def get_collection(self, name):
        self.names.append(name)
        return self.collection


def test_adapter_maps_epic_four_metadata_and_distances():
    client = Client()
    adapter = ChromaVectorSearch(client, lambda repo_id: f"repo_{repo_id}")
    hits = adapter.query("uuid", [0.1, 0.2], top_k=3)
    assert client.names == ["repo_uuid"]
    assert client.collection.query_args == {
        "query_embeddings": [[0.1, 0.2]],
        "n_results": 3,
        "include": ["documents", "metadatas", "distances"],
    }
    assert hits[0].text == "code"
    assert hits[0].path == "src/a.py"
    assert (hits[0].start_line, hits[0].end_line, hits[0].distance) == (2, 4, 0.2)
