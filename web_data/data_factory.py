import os, os.path

# Returns a dict mapping url -> data
def get_web_data(directory: str) -> dict:
    data = {}
    path, dirs, files = next(os.walk(directory))
    num_files = len(files)

    for i in range(1,num_files+1):
        f = open(directory+"/" + str(i) + ".txt", "r")
        lines = f.readlines()
        url = lines[0].strip()
        web_data = ""
        for line in lines[1:]:
            web_data += line.strip() + " "
        data[url] = web_data
    
    return data


def main():
    get_web_data("admissions")


if __name__ == "__main__":
    main()
