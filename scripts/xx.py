import sys


def delete_source_column(file_path, destination_path):
    """
    Delete the source column from the file.
    """
    # with open(file_path, "r") as f:
    #     lines = f.readlines()

    with open(destination_path, "w") as f_out:
        with open(file_path, "r") as f_in:
            for line in f_in:
                x = line.split("\t")
                x = [x[0]] + x[3:]
                new_line = "\t".join(x)
                f_out.write(new_line)


if __name__ == "__main__":
    args = sys.argv
    if args[1] == "delete_source_column":
        delete_source_column(args[2], args[3])
