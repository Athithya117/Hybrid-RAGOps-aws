from typing import Dict

def normalize_payload(chunk: Dict) -> Dict:
    """Flatten selected metadata for Qdrant payload."""
    return {
        "chunk_id":        chunk["chunk_id"],
        "document_id":     chunk["document_id"],
        "chunk_type":      chunk["chunk_type"],
        "text":            chunk["text"],

        "source.file_type": chunk["source"]["file_type"],
        "source.source_path": chunk["source"]["source_path"],
        "source.page_number": chunk["source"].get("page_number"),
        "source.time": chunk["source"].get("time"),
        "source.line_range": chunk["source"].get("line_range"),
        "source.bbox": chunk["source"].get("bbox"),

        "graph.graph_node_id": chunk["graph"].get("graph_node_id"),
        "graph.parent_id":   chunk["graph"].get("parent_id"),
        "graph.previous_id": chunk["graph"].get("previous_id"),
        "graph.next_id":     chunk["graph"].get("next_id"),

        "metadata.timestamp": chunk["metadata"].get("timestamp"),
        "metadata.tags": chunk["metadata"].get("tags", []),
        "metadata.layout_tags": chunk["metadata"].get("layout_tags", []),

        "entities": chunk.get("entities", []),
        "triplets": chunk.get("triplets", [])
    }
