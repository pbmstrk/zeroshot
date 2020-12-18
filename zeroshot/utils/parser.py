def parse_from_txt_file(f, sep="\t"):
    data = []
    with open(f) as file_obj:
        for line in file_obj:
            inter = line.split(sep)
            label = int(inter[0])
            text = inter[1].strip()
            data.append([text, label])
    return data
