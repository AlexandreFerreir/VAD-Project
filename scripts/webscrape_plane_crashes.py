import requests
from bs4 import BeautifulSoup
import csv
import time
import random
import re

# URL base e cabeçalhos
BASE_URL = "https://www.planecrashinfo.com/"
DATABASE_URL = BASE_URL + "database.htm"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/115.0 Safari/537.36")
}

def get_year_links():
    """
    Acessa a página principal do banco de dados e extrai os links referentes a cada ano.
    Os links seguem o padrão 'YYYY/YYYY.htm'.
    """
    response = requests.get(DATABASE_URL, headers=HEADERS)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    
    pattern = re.compile(r'(\d{4})/(\1)\.htm')
    year_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if pattern.search(href):
            # Completa o link se for relativo
            full_url = href if href.startswith("http") else BASE_URL + href
            year = a.get_text(strip=True)
            year_links.append((year, full_url))
    return year_links

def parse_detailed_page(url):
    """
    Acessa a página de detalhes de um acidente e extrai os campos informados.
    Agora, localiza a tabela que contém os dados (os dados estão organizados
    em linhas com duas células: campo e valor) e os mapeia para a estrutura desejada.
    """
    print(f"Acessando página de detalhes: {url}")
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Campos padrão (valores desconhecidos)
    details = {
        "Date": "?",
        "Time": "?",
        "Location": "?",
        "Operator": "?",
        "Flight #": "?",
        "Route": "?",
        "AC Type": "?",
        "Registration": "?",
        "cn / ln": "?",
        "Aboard": "?",
        "Fatalities": "?",
        "Ground": "0",
        "Summary": "?"
    }
    
    # Procura a primeira tabela que provavelmente contém os detalhes
    table = soup.find("table")
    if table:
        rows = table.find_all("tr")
        # Itera sobre as linhas da tabela (ignora o cabeçalho se necessário)
        for row in rows:
            cells = row.find_all("td")
            if len(cells) >= 2:
                # extrai o texto, incluindo eventuais quebras de linha
                raw = cells[0].get_text(separator=" ", strip=True)
                # substitui qualquer sequência de whitespace (espaços, tabs, \n…) por um único espaço
                field_text = " ".join(raw.split())
                if field_text.endswith(":"):
                    field_text = field_text[:-1]
                value_text = cells[1].get_text(separator=" ", strip=True)
                if field_text in details:
                    details[field_text] = value_text
    else:
        # Fallback: se não encontrar a tabela, processa o texto inteiro
        detail_text = soup.get_text(separator="\n")
        for line in detail_text.splitlines():
            if ':' in line:
                parts = line.split(":", 1)
                field = parts[0].strip()
                value = parts[1].strip()
                if field in details:
                    details[field] = value
                    
    return details

def parse_year_page(year, url):
    """
    Para a página de um ano, extrai cada linha da tabela de acidentes.
    Se a célula da data tiver um link para a página de detalhes, acessa-o para coletar informações.
    """
    print(f"Processando dados do ano: {year}")
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    
    accidents = []
    table = soup.find("table")
    if table:
        rows = table.find_all("tr")
        # Pula a linha de cabeçalho
        for row in rows[1:]:
            cols = row.find_all("td")
            if len(cols) < 4:
                continue

            date_cell = cols[0]
            detail_link_tag = date_cell.find("a")
            if detail_link_tag:
                href = detail_link_tag["href"].strip()
                # Se o link não contiver o diretório do ano, adiciona-o
                if not href.startswith(f"{year}/"):
                    full_link = BASE_URL + f"{year}/{href}"
                else:
                    full_link = href if href.startswith("http") else BASE_URL + href
                try:
                    record = parse_detailed_page(full_link)
                    #print(f"Dados extraídos de {full_link}: {record}")
                except Exception as e:
                    print(f"Erro ao acessar detalhes em {full_link}: {e}")
                    record = {}
                # Se o campo "Date" não foi extraído, define-o a partir do resumo da listagem
                if record.get("Date", "?") in ["", "?"]:
                    record["Date"] = f"{date_cell.get_text(strip=True)} {year}"
            else:
                # Caso não haja link de detalhes, extrai os dados resumidos da listagem
                date_str = date_cell.get_text(strip=True)
                loc_op = cols[1].get_text(separator="\n", strip=True).split("\n")
                location = loc_op[0] if len(loc_op) >= 1 else "?"
                operator = loc_op[1] if len(loc_op) >= 2 else "?"
                ac_reg = cols[2].get_text(separator="\n", strip=True).split("\n")
                ac_type = ac_reg[0] if len(ac_reg) >= 1 else "?"
                registration = ac_reg[1] if len(ac_reg) >= 2 else "?"
                fatalities = cols[3].get_text(strip=True)
                record = {
                    "Date": f"{date_str} {year}",
                    "Time": "?",
                    "Location": location,
                    "Operator": operator,
                    "Flight #": "?",
                    "Route": "?",
                    "AC Type": ac_type,
                    "Registration": registration,
                    "cn / ln": "?",
                    "Aboard": "?",
                    "Fatalities": fatalities,
                    "Ground": "0",
                    "Summary": "?"
                }
            accidents.append(record)
            # Delay para imitar o comportamento humano e reduzir o risco de bloqueio
            time.sleep(random.uniform(1, 3))
    else:
        print(f"Tabela não encontrada para o ano {year}.")
    return accidents

def main():
    all_accidents = []
    
    # Extrai os links dos anos
    year_links = get_year_links()
    print(f"Foram encontrados {len(year_links)} links de anos.")
    
    # Processa cada ano
    for year, url in year_links:
        accidents = parse_year_page(year, url)
        all_accidents.extend(accidents)
    
    print(f"Foram encontrados {len(all_accidents)} acidentes no total.")
    # Define os cabeçalhos do CSV conforme a estrutura requerida
    fieldnames = [
        "Date", "Time", "Location", "Operator", "Flight #", "Route",
        "AC Type", "Registration", "cn / ln", "Aboard", "Fatalities",
        "Ground", "Summary"
    ]
    
    output_csv = "data\\crashes_data\\plane_crash_data.csv"
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for record in all_accidents:
            writer.writerow(record)
    
    print(f"Dados detalhados salvos em {output_csv}")

if __name__ == "__main__":
    main()