import requests
from bs4 import BeautifulSoup

# Liste des URLs à scraper
urls = [
    "https://www.nature.com/articles/d41586-020-00502-w",
    "https://www.nejm.org/doi/full/10.1056/NEJMoa2033700?query-featured_coronavirus=",
    "https://www.nejm.org/doi/full/10.1056/NEJMoa2030340?query=featured_coronavirus",
    "https://www.nejm.org/doi/full/10.1056/NEJMoa2035002?query-featured_coronavirus=",
    "https://www.nejm.org/doi/full/10.1056/NEJMoa2029849?query=featured_coronavirus",
    "https://www.nejm.org/doi/full/10.1056/NEJMpv2035416?query=featured_coronavirus",
    "https://www.thelancet.com/journals/lanrhe/article/PIIS2665-9913(21)00007-2/fulltext",
    "https://www.thelancet.com/journals/lanres/article/PIIS2213-2600(21)00025-4/fulltext",
    "https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(20)32656-8/fulltext",
    "https://science.sciencemag.org/content/early/2021/01/11/science.abe6522"
]

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

# Itération sur chaque URL pour télécharger et sauvegarder le contenu
for i, url in enumerate(urls, start=1):
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extraction du texte des divs avec role="paragraph"
    texte_article = ' '.join(div.get_text() for div in soup.find_all('div', {'role': 'paragraph'}))
    texte_article += ' '.join(div.get_text() for div in soup.find_all('div', {'class': 'section-paragraph'}))


    # Création du fichier pour stocker l'article, numéroté de 1 à 10
    nom_fichier = f"{i}.txt"
    with open(nom_fichier, "w", encoding='utf-8') as fichier:
        fichier.write(texte_article)
    print(f"Le texte de l'article a été sauvegardé dans : {nom_fichier}")
