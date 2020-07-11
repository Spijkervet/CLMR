
def label_to_tag(list_of_tags, label):
    with open(list_of_tags, "r") as f:
        tags = f.readlines()
    return tags[label]