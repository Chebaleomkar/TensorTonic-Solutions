def jaccard_similarity(set_a, set_b):
    """
    Compute the Jaccard similarity between two item sets.
    """
    # Use variable names that match the type (sets)
    s1 = set(set_a)
    s2 = set(set_b)

    # Calculate union first to check for zero
    union_len = len(s1 | s2)
    
    if union_len == 0:
        return 0.0  # Handle the empty set case

    intersection_len = len(s1 & s2)

    return intersection_len / union_len
