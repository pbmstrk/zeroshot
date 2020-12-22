def parse_from_txt_file(f, label_map=None, sep="\t"):
    data = []
    with open(f) as file_obj:
        for line in file_obj:
            inter = line.split(sep)
            if label_map:
                label = label_map[inter[0]]
            else:
                label = inter[0]
            text = inter[1].strip()
            data.append([text, label])
    return data
