import requests
from bs4 import BeautifulSoup
from time import sleep

'''
IMPORTANT. To use this, simply run python3 web_scrape.py
To pull a specific subdomain, you need to create a folder to store the 
text.

Next, change test_url to the correct subdomain url, and chain file_path
in the main() function to the folder you created.
'''

test_url = "https://covid19.illinois.edu"
dict_href_links = {}
set_links_to_visit = set()

def get_data(url):
    r = requests.get(url)
    return r.text

def scrape_and_save(url, file_path="covid19/", filename=0):
    html_data = get_data(url)
    soup = BeautifulSoup(html_data, "html.parser")
    heading_tags = ["h1", "h2", "h3", "p"]
    paragraph_data = soup.find_all(heading_tags)

    f = open(file_path+str(filename)+".txt", "w")
    f.write(url + "\n")
    if soup.title:
        f.write(soup.title.string + "\n")
    for p in paragraph_data:
        # print(p.text)
        f.write(str(p.text) + "\n")
    # print(paragraph_data)

def get_links(website_link):
    html_data = get_data(website_link)
    soup = BeautifulSoup(html_data, "html.parser")
    list_links = []
    for link in soup.find_all("a", href=True):
        # Append to list if new link contains original link
        if str(link["href"]).startswith((str(website_link))):
            print("link =", link["href"])
            list_links.append(link["href"])
            set_links_to_visit.add(link["href"])
        elif str(link["href"]).startswith("//"):
            continue
        # Include all href that do not start with website link but with "/"
        elif str(link["href"]).startswith("/"):
            #print("----Test-------")
            if link["href"] not in dict_href_links:
                # print(link["href"])
                dict_href_links[link["href"]] = None
                link_with_www = website_link +"/"+ link["href"][1:]
                print("adjusted link =", link_with_www)
                list_links.append(link_with_www)
                set_links_to_visit.add(link_with_www)
        elif "http" not in str(link["href"]) and "html" in str(link["href"]):
            if link["href"] not in dict_href_links:
                # print(link["href"])
                dict_href_links[link["href"]] = None
                link_with_www = website_link +"/"+ link["href"]
                print("adjusted link 2 =", link_with_www)
                list_links.append(link_with_www)
                set_links_to_visit.add(link_with_www)
            

                
    # Convert list of links to dictionary and define keys as the links
    dict_links = dict.fromkeys(list_links, False)
    return dict_links

def main():
    get_links(test_url)
    # print(set_links_to_visit)
    ctr = 1
    for link in set_links_to_visit:
        scrape_and_save(link, file_path="covid19/", filename=ctr)
        sleep(0.5)
        ctr += 1
    scrape_and_save(test_url, filename=1)


if __name__ == "__main__":
    main()