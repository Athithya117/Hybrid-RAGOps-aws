from refined.inference.processor import Refined

# Initialize the model
refined = Refined.from_pretrained(
    model_name='wikipedia_model_with_numbers',  # Choose model variant
    entity_set="wikipedia",                     # Use Wikipedia entity set (~6M entities)
    download_files=True,                        # Download model files on first run
    use_precomputed_descriptions=True           # Speed up inference
)

# Process some text
text = "Apple CEO Tim Cook announced new products at the event in California yesterday."
spans = refined.process_text(text)

# Print results
for span in spans:
    print(f"Text: {span.text}")
    print(f"Entity: {span.entity.wikipedia_entity_title} ({span.entity.wikidata_entity_id})")
    print(f"Type: {span.coarse_type}")
    print("---")
    