# in config.py (extend existing Settings)
    # Vector index specifics for ArangoDB
    ARANGO_VECTOR_FIELD     = os.getenv("ARANGO_VECTOR_FIELD", "embedding")
    ARANGO_VECTOR_METRIC    = os.getenv("ARANGO_VECTOR_METRIC", "cosine")  # "cosine" | "l2"
    ARANGO_VECTOR_INDEX_NAME = os.getenv("ARANGO_VECTOR_INDEX_NAME", "vec_idx")
    # Optionally provide JSON string for additional params (nLists, factory, defaultNProbe)
    # Example: '{"nLists": 256, "factory": "IVF256,PQ64", "defaultNProbe": 10}'
    ARANGO_VECTOR_PARAMS_JSON = os.getenv("ARANGO_VECTOR_PARAMS_JSON", "")
    @property
    def ARANGO_VECTOR_PARAMS(self):
        if self.ARANGO_VECTOR_PARAMS_JSON:
            try:
                return json.loads(self.ARANGO_VECTOR_PARAMS_JSON)
            except Exception:
                return {}
        return {}
