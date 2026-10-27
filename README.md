# Dokumentacja projektu: rozpoznawanie dań i składników ze zdjęć

**Wersja dokumentacji:** 1.0 (27.10.2026)
**Zbiór danych:** [MM-Food-100K](https://huggingface.co/datasets/Humanbased-AI/MM-Food-100K) (Hugging Face)
**Język / środowisko:** Python, PyTorch, scikit-learn, matplotlib
**Kontekst:** projekt z przedmiotu *Uczenie Maszyn*, ścieżka aplikacyjna (implementacja prototypu systemu)

---

## Spis treści

1. [Opis projektu](#1-opis-projektu)
2. [Metody nauczania wykorzystane w modelu](#2-metody-nauczania-wykorzystane-w-modelu)
3. [Struktura projektu (drzewo plików)](#3-struktura-projektu-drzewo-plików)
4. [Przepływ danych](#4-przepływ-danych)
5. [Dane](#5-dane)
6. [Opis modułów i skryptów](#6-opis-modułów-i-skryptów)
7. [Wymagania i instalacja](#7-wymagania-i-instalacja)
8. [Uruchomienie krok po kroku](#8-uruchomienie-krok-po-kroku)
9. [Parametry konfiguracyjne](#9-parametry-konfiguracyjne)
10. [Wyniki i metryki](#10-wyniki-i-metryki)
11. [Użycie programistyczne (API)](#11-użycie-programistyczne-api)
12. [Zgodność z wytycznymi przedmiotu](#12-zgodność-z-wytycznymi-przedmiotu)
13. [Ograniczenia i znane problemy](#13-ograniczenia-i-znane-problemy)
14. [Rozwiązywanie problemów](#14-rozwiązywanie-problemów)
15. [Możliwe rozszerzenia](#15-możliwe-rozszerzenia)
16. [Bibliografia](#16-bibliografia)

---

## 1. Opis projektu

System na podstawie **jednego zdjęcia posiłku** zwraca:

| Wyjście | Opis | Przykład |
|---|---|---|
| **Danie** | trzy najbardziej prawdopodobne dania z prawdopodobieństwami | `Pho (0.41)`, `Noodle Soup (0.22)`, … |
| **Składniki** | lista rozpoznanych składników z prawdopodobieństwami | `noodle`, `beef`, `green onion` |
| **Wartości odżywcze** | szacunek kalorii, białka, tłuszczu, węglowodanów i łącznej wagi porcji | `calories_kcal: 450.0` |

Model jest **własną siecią konwolucyjną trenowaną od zera**. Nie korzysta z żadnych pretrenowanych wag ani modeli. Z Hugging Face pobierany jest wyłącznie zbiór danych.

Projekt składa się z trzech etapów uruchamianych osobnymi skryptami: przygotowanie danych, trening z oceną, prototyp aplikacji.

---

## 2. Metody nauczania wykorzystane w modelu

### 2.1. Podsumowanie

Całość jest realizowana jako **nadzorowane uczenie wielozadaniowe (ang. *supervised multi-task learning*) z twardym współdzieleniem parametrów (ang. *hard parameter sharing*)** (Caruana, 1997). Jedna sieć konwolucyjna (wspólny „korpus") produkuje wektor cech zdjęcia, a trzy osobne „głowice" wykonują trzy różne zadania uczenia maszynowego:

| Głowica | Zadanie | Nazwa metody | Funkcja aktywacji wyjścia | Funkcja straty |
|---|---|---|---|---|
| **Danie** | klasyfikacja jednoetykietowa wieloklasowa (*multiclass classification*) | klasyfikator softmax z wygładzaniem etykiet (*label smoothing*) | softmax | entropia krzyżowa (*categorical cross-entropy*) |
| **Składniki** | klasyfikacja wieloetykietowa (*multi-label classification*) | **Binary Relevance** zrealizowana w sieci neuronowej | sigmoida (osobna dla każdego składnika) | binarna entropia krzyżowa (*binary cross-entropy*) |
| **Wartości odżywcze** | regresja wielowyjściowa (*multi-target / multi-output regression*) | jedna sieć o wielu wyjściach ciągłych | brak (wartości surowe) | strata Hubera (*Huber / smooth L1*) |

Funkcja straty całego modelu jest ważoną sumą trzech strat:

```
L = 1 · L_danie + 5 · L_składniki + 1 · L_odżywianie
```

Waga 5 dla składników kompensuje fakt, że uśredniona binarna entropia krzyżowa po wielu etykietach (większość z nich to „nieobecny") ma małą wartość i bez wzmocnienia byłaby zdominowana przez pozostałe straty.

### 2.2. Danie: klasyfikacja wieloklasowa

- **Problem:** każde zdjęcie ma dokładnie jedno poprawne danie spośród *N* klas.
- **Metoda:** głowica liniowa zwraca po jednym wyniku na danie, a funkcja *softmax* zamienia je w prawdopodobieństwa sumujące się do 1. Predykcją jest danie o największym prawdopodobieństwie, a w aplikacji pokazywane są trzy najlepsze (*top-3*).
- **Strata:** entropia krzyżowa. Karze niskie prawdopodobieństwo przypisane prawdziwemu daniu.
- **Regularyzacja:** wygładzanie etykiet (*label smoothing*, ze współczynnikiem 0,1. Prawdziwa klasa dostaje cel ok. 0,9 zamiast 1,0, co zmniejsza nadmierną pewność modelu. Wygładzanie jest stosowane tylko w trakcie uczenia, a nie przy liczeniu straty walidacyjnej.

### 2.3. Składniki: klasyfikacja wieloetykietowa (Binary Relevance)

- **Problem:** każde zdjęcie ma zbiór składników o zmiennej liczebności (od jednego do wielu). W terminologii pracy Gibaja i Ventura (2015) jest to zadanie *Multilabel Classification* (MLC), czyli nauczenie funkcji *h: X → 2^L*, która zwraca zbiór etykiet trafnych.
- **Metoda:** **Binary Relevance (BR)**. Problem wieloetykietowy rozkładany jest na niezależne decyzje binarne, po jednej dla każdego składnika („czy składnik *x* jest na zdjęciu?"). Każdy składnik ma własne wyjście sigmoidalne, więc prawdopodobieństwa są niezależne i nie sumują się do 1.
- **Realizacja:** w klasycznym BR trenuje się osobny klasyfikator dla każdej etykiety. W tym projekcie wszystkie decyzje binarne są wyjściami **jednej sieci neuronowej** (do 150 sigmoid), współdzielącej warstwy konwolucyjne, i uczą się jednocześnie. Funkcja straty (suma binarnych entropii krzyżowych) i reguła decyzyjna są równoważne BR, natomiast strukturalnie jest to sieć neuronowa (w tej taksonomii *algorithm adaptation*).
- **Zależności między etykietami:** BR zakłada niezależność etykiet. Model nie ma jawnego mechanizmu zależności. Wspólne cechy obrazu mogą je uwzględniać jedynie pośrednio.
- **Reguła decyzyjna:** sieć zwraca prawdopodobieństwa, a jeden **globalny próg** *t* zamienia je na zbiór składników. Próg dobierany jest na zbiorze walidacyjnym jako ten, który maksymalizuje *micro-F1* (siatka 0,05 – 0,60 co 0,025). Jeśli żaden składnik nie przekroczy progu, zwracany jest ten o najwyższym prawdopodobieństwie, więc lista nigdy nie jest pusta.

### 2.4. Wartości odżywcze: regresja wielowyjściowa

- **Problem:** przewidzenie pięciu wartości ciągłych: `calories_kcal`, `protein_g`, `fat_g`, `carbohydrate_g` oraz `total_weight_g` (suma wag składników z kolumny `portion_size`, dodana w tym projekcie).
- **Metoda:** jedna sieć z pięcioma wyjściami liniowymi.
- **Przetwarzanie celów:** transformacja `log1p`, a następnie standaryzacja (średnia 0, odchylenie 1). Dzięki temu błąd jest porównywalny dla małych i dużych porcji, a rozkład (prawoskośny) jest bliższy normalnemu. Predykcje są przeliczane z powrotem funkcją `expm1` i obcinane do wartości nieujemnych.
- **Strata:** funkcja Hubera (*smooth L1*), czyli kwadratowa dla małych błędów, a liniowa dla dużych. Jest odporna na wartości odstające, co ma znaczenie, bo dane odżywcze w zbiorze są szacunkami adnotatorów.
- **Brakujące wartości:** jeśli któryś cel jest nieznany (`NaN`, np. brak wagi porcji), jest maskowany i nie wchodzi do straty.

### 2.5. Architektura sieci (wspólny korpus)

Sieć `FoodNet` (plik `foodrec/foodnet.py`) to niewielka sieć rezydualna w stylu ResNet, inicjalizowana losowo:

```
Wejście: obraz 3 × 128 × 128, wartości w [-1, 1]
 │
 ├─ Warstwa wejściowa: Conv 3×3 (krok 2) → BatchNorm → ReLU                 [w filtrów]
 ├─ Etap 1: 2 × ResBlock                                                   [w   filtrów]
 ├─ Etap 2: 2 × ResBlock (pierwszy z krokiem 2)                            [2w  filtrów]
 ├─ Etap 3: 2 × ResBlock (pierwszy z krokiem 2)                            [4w  filtrów]
 ├─ Etap 4: 2 × ResBlock (pierwszy z krokiem 2)                            [8w  filtrów]
 ├─ Global Average Pooling  → wektor o długości 8w
 ├─ Warstwa wspólna: Linear(8w → 8w) → ReLU → Dropout(0,3)
 │
 ├─ Głowica „danie":      Linear(8w → liczba_dań)
 ├─ Głowica „składniki":  Linear(8w → liczba_składników)
 └─ Głowica „odżywianie": Linear(8w → 5)
```

Domyślnie `w = 32`, więc liczba filtrów wynosi kolejno 32, 64, 128 i 256, co daje szacunkowo ok. 3 mln parametrów.

Użyte elementy:

| Element | Rola |
|---|---|
| **Konwolucja** | wykrywa wzorce w obrazie (krawędzie → tekstury → kształty) |
| **Normalizacja wsadowa**  | stabilizuje i przyspiesza uczenie |
| **ReLU** | nieliniowość |
| **Blok rezydualny** (*ResBlock*) | dwie konwolucje 3×3 oraz połączenie skrótowe (suma wejścia i wyjścia), co ułatwia przepływ gradientu |
| **Global Average Pooling** | uśrednia każdy filtr po całym obrazie i daje jeden wektor opisujący zdjęcie |
| **Dropout**  | losowo wyłącza 30% neuronów w trakcie uczenia, co ogranicza przeuczenie |

### 2.6. Procedura uczenia

| Element | Opis |
|---|---|
| **Propagacja wsteczna** na mini-wsadach | gradient straty liczony dla 64 obrazów naraz |
| **Optymalizator AdamW**  | adaptacyjny krok dla każdej wagi oraz rozdzielony zanik wag (*weight decay* = 0,05) |
| **Harmonogram One-Cycle**  | współczynnik uczenia rośnie do wartości szczytowej (`lr` = 2·10⁻³), a potem maleje niemal do zera |
| **Augmentacja danych** | losowo w każdym wsadzie: skala 0,7–1,0, obrót ±0,17 rad (≈ ±10°), przesunięcie, odbicie poziome, zmiana kontrastu (0,8–1,2) i jasności (±0,15) |
| **Wczesne zatrzymanie** | jeśli strata walidacyjna nie poprawia się przez 6 epok z rzędu, uczenie się kończy, a przywracane są wagi z najlepszej epoki |
| **Mieszana precyzja** (*AMP*) | liczby 16-bitowe tam, gdzie to bezpieczne.
| **Dobór progu składników** | po uczeniu, na zbiorze walidacyjnym (patrz 2.3) |

Domyślnie uczenie trwa maksymalnie **30 epok** (epoka = jedno przejście po całym zbiorze uczącym). Wczesne zatrzymanie może je skrócić. Ponieważ harmonogram One-Cycle jest planowany na pełne 30 epok, wcześniejsze zatrzymanie przerywa go, zanim współczynnik uczenia zdąży w pełni zmaleć, co jest zachowaniem normalnym.

### 2.7. Podział danych i ocena

- **Podział „holdout"** 70% / 15% / 15% (uczący / walidacyjny / testowy), **stratyfikowany względem dania** (każde danie ma podobny udział w każdej części). W klasyfikacji wieloetykietowej stratyfikacja względem zbiorów etykiet jest problemem otwartym, więc nie jest stosowana dla składników.
- **Zbiór walidacyjny** służy do wczesnego zatrzymania i doboru progu składników.
- **Zbiór testowy** jest używany **jednorazowo**, na końcu `02_train.py`, i nie wpływa na żadne decyzje.
- Metryki są opisane w [rozdziale 10](#10-wyniki-i-metryki).

---

## 3. Struktura projektu (drzewo plików)

```
food_recognition/
├── README.md                     # skrócony opis i szybki start
├── DOKUMENTACJA.md               # niniejszy dokument
├── requirements.txt              # zależności Pythona
│
├── foodrec/                      # pakiet z logiką (każdy moduł = osobna odpowiedzialność)
│   ├── __init__.py               # oznacza katalog jako pakiet Pythona
│   ├── config.py                 # ścieżki, stałe (zbiór HF, rozmiar obrazu, ziarno, cele regresji)
│   ├── data.py                   # wczytanie z HF, czyszczenie, parsowanie pól, pobieranie obrazów,
│   │                             #   podział train/val/test, konwersja obrazów do tablic
│   ├── foodnet.py                # architektura sieci, augmentacja, FoodNetEstimator (API scikit-learn)
│   ├── evaluation.py             # metryki (scikit-learn) i wykresy (matplotlib)
│   └── pipeline.py               # FoodRecognizer: zdjęcie → danie / składniki / wartości odżywcze
│
├── scripts/                      # skrypty uruchamiane z linii poleceń (kolejność = numer)
│   ├── 01_prepare_data.py        # etap 1: dane z HF → metadata.parquet + obrazy
│   ├── 02_train.py               # etap 2: trening, ocena na teście, wykresy, zapis modelu
│   └── 03_app.py                 # etap 3: prototyp (Gradio) lub predykcja z CLI
│
└── artifacts/                    # TWORZONY AUTOMATYCZNIE podczas uruchamiania (nie ma go w repozytorium)
    ├── metadata.parquet          # oczyszczona tabela: etykiety, ścieżki obrazów, kolumna `split`
    ├── images/                   # pobrane obrazy (nazwa = skrót MD5 adresu URL, format JPEG)
    ├── images_128.npz            # cache obrazów jako tablica uint8 (N × 128 × 128 × 3)
    ├── food_recognizer.joblib    # zapisany, gotowy system (sieć + nazwy klas)
    └── results/
        ├── test_report.json          # metryki na zbiorze testowym + parametry uczenia
        ├── training_curves.png       # krzywe uczenia (strata, dokładność dania na walidacji)
        ├── test_confusion_dishes.png # macierz pomyłek dla 20 najczęstszych dań
        ├── test_ingredient_f1.png    # F1 dla 30 najczęstszych składników
        └── test_nutrition_scatter.png# wykresy: wartość prawdziwa vs przewidywana
```

---

## 4. Przepływ danych

```
Hugging Face (MM-Food-100K)
        │  data.load_raw()
        ▼
   surowa tabela (image_url, dish_name, ingredients, portion_size, nutritional_profile, …)
        │  data.build_table()   ← filtrowanie, normalizacja, top-N dań i składników
        ▼
   oczyszczona tabela
        │  data.download_images()   ← pobranie obrazów (martwe linki pomijane)
        │  data.assign_split()      ← kolumna `split` = train / val / test
        ▼
   artifacts/metadata.parquet + artifacts/images/          (skrypt 01)
        │  data.load_or_build_array()   ← obrazy → tablica uint8 (cache .npz)
        ▼
   FoodNetEstimator.fit(train, eval_set=val)               (skrypt 02)
        │  + ocena na teście, wykresy, raport JSON
        ▼
   artifacts/food_recognizer.joblib
        │  FoodRecognizer.load()
        ▼
   aplikacja Gradio / CLI: zdjęcie → danie, składniki, wartości odżywcze   (skrypt 03)
```

---

## 5. Dane

### 5.1. Zbiór MM-Food-100K

Zbiór zawiera ok. 100 tys. wierszy i **tylko podział `train`** (brak oficjalnego podziału walidacyjnego i testowego). Obrazy nie są częścią zbioru, są dostępne pod adresami URL w kolumnie `image_url`.

| Kolumna | Opis | Użycie w projekcie |
|---|---|---|
| `image_url` | link do obrazu | pobieranie obrazów |
| `dish_name` | nazwa dania | etykieta klasyfikacji (po normalizacji) |
| `ingredients` | lista składników (tekst JSON) | etykiety wieloetykietowe |
| `portion_size` | wagi składników (tekst JSON) | suma wag → cel regresji `total_weight_g` |
| `nutritional_profile` | kalorie, białko, tłuszcz, węglowodany (tekst JSON) | cele regresji |
| `food_prob` | prawdopodobieństwo, że obraz zawiera jedzenie | filtrowanie |
| `camera_or_phone_prob`, `online_download_prob` | źródło zdjęcia | opcjonalne filtrowanie (`--camera-only`) |
| `food_type`, `cooking_method` | rodzaj jedzenia, sposób przygotowania | **nieużywane** (kolumny są zachowywane w tabeli) |

### 5.2. Przygotowanie danych (`data.build_table`)

Kroki wykonywane w tej kolejności:

1. Usunięcie wierszy bez `image_url` lub `dish_name` oraz duplikatów adresów URL.
2. Filtr `food_prob ≥ --min-food-prob` (domyślnie 0,5).
3. Opcjonalnie (`--camera-only`): tylko zdjęcia, dla których `camera_or_phone_prob ≥ online_download_prob`.
4. Normalizacja nazwy dania: małe litery, usunięcie nawiasów i znaków specjalnych, ujednolicenie spacji. Powtarzające się nazwy łączą się w jedną klasę.
5. Wybór dań: zostają dania mające co najmniej `--min-per-class` przykładów, ograniczone do `--top-dishes` najczęstszych.
6. Ograniczenie liczby przykładów na danie do `--max-per-class` (losowe, z ustalonym ziarnem), co zmniejsza niezbalansowanie klas i rozmiar pobierania.
7. Składniki: wczytanie listy (JSON, lista Pythona lub tekst rozdzielony przecinkami), normalizacja (małe litery, bez znaków specjalnych, ostatnie słowo w liczbie pojedynczej, np. `green onions` → `green onion`, `tomatoes` → `tomato`), usunięcie powtórzeń w wierszu, ograniczenie do słownika `--top-ingredients` najczęstszych składników. Wiersze bez żadnego składnika są odrzucane.
8. Cele regresji: wczytanie wartości odżywczych (wiersze bez nich są odrzucane; dodatkowo `0 < kalorie < 5000`) oraz obliczenie `total_weight_g` jako sumy wag z `portion_size`. Obsługiwane jednostki to `g`, `kg`, `ml` i `l` (1 ml liczone jak 1 g). Pozycje bez takiej jednostki (np. „10 pieces") są pomijane, a gdy nie ma żadnej, waga to `NaN` i jest maskowana w stracie.

Następnie `download_images` pobiera obrazy (wielowątkowo), skaluje do maks. `--max-side` pikseli i zapisuje jako JPEG. Wiersze z niedostępnym obrazem są usuwane. Na końcu `assign_split` wykonuje stratyfikowany podział 70/15/15 (dania z mniej niż 10 przykładami są pomijane).

### 5.3. Przygotowanie obrazów dla sieci (`data.to_array`)

Skalowanie krótszego boku do 128 px (interpolacja bikubiczna), kadr środkowy do kwadratu 128×128, zapis jako `uint8`. Przed podaniem do sieci wartości są skalowane do przedziału [-1, 1].

---

## 6. Opis modułów i skryptów

### 6.1. Pakiet `foodrec`

#### `config.py`
Stałe i ścieżki: `HF_DATASET`, `IMG_SIZE` (128), `SEED` (42), `NUTRITION_COLS`, `TARGET_COLS` (kolumny odżywcze + `total_weight_g`), ścieżki do `artifacts/`. Funkcja `ensure_dirs()` tworzy potrzebne katalogi.

#### `data.py`

| Funkcja | Opis |
|---|---|
| `load_raw()` | pobiera tabelę z Hugging Face (`datasets.load_dataset`) jako `DataFrame` |
| `normalize_text`, `normalize_ingredient` | normalizacja tekstu, singularyzacja składnika |
| `parse_ingredients`, `parse_nutrition`, `parse_total_weight` | parsowanie pól tekstowych (obsługa JSON, list Pythona, tekstu z przecinkami, brakujących wartości) |
| `build_table(raw, …)` | czyszczenie i ograniczenie do najczęstszych klas i składników (patrz 5.2) |
| `download_images(df, …)` | pobieranie obrazów do `artifacts/images`, zwraca tylko wiersze z poprawnym plikiem |
| `assign_split(df, …)` | dodaje kolumnę `split` (stratyfikowany podział) |
| `to_array(img, size)` | ścieżka lub obiekt PIL → tablica `uint8` (skalowanie + kadr środkowy) |
| `load_or_build_array(paths, size)` | tablica wszystkich obrazów z cache'em `artifacts/images_<size>.npz` (klucz = skrót listy ścieżek) |
| `load_metadata()` | wczytuje `metadata.parquet` |
| `regression_targets(df)` | macierz celów regresji (brakujące = `NaN`) |

#### `foodnet.py`

| Element | Opis |
|---|---|
| `ResBlock` | blok rezydualny (conv–BN–ReLU–conv–BN + skrót) |
| `FoodNet` | sieć z trzema głowicami (patrz 2.5) |
| `augment(x)` | augmentacja wsadowa na tensorach (skala, obrót, odbicie, przesunięcie, jasność, kontrast) |
| `FoodNetEstimator` | **estymator zgodny z API scikit-learn** (`BaseEstimator`): `fit(X, y, eval_set)`, `predict`, `predict_proba`, `predict_ingredient_proba`, `predict_ingredients`, `predict_nutrition`, `get_params`/`set_params` |

Format danych `FoodNetEstimator.fit`:
- `X`: `numpy.uint8` o kształcie `(N, H, W, 3)`,
- `y`: słownik `{"dish": int[N], "ingredients": 0/1[N, K], "nutrition": float[N, 5]}` (`NaN` dozwolone w `nutrition`),
- `eval_set`: opcjonalna para `(X_val, y_val)` używana do wczesnego zatrzymania i doboru progu.

Po uczeniu dostępne są atrybuty: `history_` (lista słowników z przebiegiem epok), `ingredient_threshold_`, `classes_`, `model_`.

#### `evaluation.py`
Funkcje metryk (`dish_metrics`, `ingredient_metrics`, `regression_metrics`) oparte na `sklearn.metrics` oraz wykresy matplotlib (`plot_history`, `plot_confusion`, `plot_ingredient_f1`, `plot_regression_scatter`).

#### `pipeline.py`
Klasa `FoodRecognizer` łączy wytrenowany estymator z nazwami klas i przetwarzaniem obrazów. Metody: `predict(images, top_k=3)`, `save(path)` oraz `load(path)` (serializacja `joblib`).

### 6.2. Skrypty

| Skrypt | Zadanie | Wejście | Wyjście |
|---|---|---|---|
| `01_prepare_data.py` | pobranie i przygotowanie danych | Hugging Face | `metadata.parquet`, `images/` |
| `02_train.py` | trening, ocena, wykresy, zapis modelu | `metadata.parquet`, `images/` | `food_recognizer.joblib`, `results/*` |
| `03_app.py` | prototyp aplikacji lub predykcja z CLI | `food_recognizer.joblib` | interfejs w przeglądarce / wynik JSON |

---

## 7. Wymagania i instalacja

### 7.1. Wymagania

- **Python** 3.9 lub nowszy (składnię sprawdzono na Pythonie 3.13).
- **Pakiety** (plik `requirements.txt`): `numpy`, `pandas`, `pyarrow`, `scikit-learn`, `matplotlib`, `datasets`, `requests`, `pillow`, `tqdm`, `joblib`, `torch`, `gradio`.
- **Zalecane minimalne wersje:** `scikit-learn ≥ 1.3`, `torch ≥ 2.3` (kod używa `torch.amp.GradScaler("cuda", …)`), `gradio ≥ 5.0` (kod używa argumentu `flagging_mode`). Plik `requirements.txt` w projekcie nie przypina wersji `torch` i `gradio`, więc przy starszych wersjach należy je zaktualizować.
- **Połączenie z internetem** podczas etapu 1 (zbiór z Hugging Face i obrazy z serwera `file.b18a.io`).
- **Pamięć RAM:** tablica obrazów zajmuje ok. 49 KB na obraz przy 128×128 (np. 30 tys. obrazów to ok. 1,5 GB, a przy maksymalnych 40 tys. ok. 2 GB). Przy 64×64 jest to czterokrotnie mniej.
- **Miejsce na dysku:** obrazy (do 384 px dłuższego boku) zajmują zwykle od kilku do kilkunastu GB dla pełnego zbioru, a mniej przy mniejszych ustawieniach.
- **Karta graficzna:** zalecana NVIDIA (CUDA). Bez niej trening działa na CPU, ale jest znacznie wolniejszy.

### 7.2. Instalacja

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Dla treningu na GPU NVIDIA najpierw zainstaluj wersję PyTorch z CUDA według instrukcji na stronie pytorch.org, a dopiero potem wykonaj `pip install -r requirements.txt`. W przeciwnym razie `pip` zainstaluje wersję CPU.

---

## 8. Uruchomienie krok po kroku

Wszystkie polecenia uruchamiaj z głównego katalogu projektu (`food_recognition/`).

### 8.1. Szybki test całego przepływu (kilkanaście minut)

```bash
python scripts/01_prepare_data.py --top-dishes 20 --max-per-class 100
python scripts/02_train.py --epochs 2 --width 16 --img-size 64
python scripts/03_app.py
```

Służy wyłącznie do sprawdzenia, czy wszystko działa i ile trwa jedna epoka. Wyniki tak uczonego modelu nie mają wartości merytorycznej.

### 8.2. Pełny przebieg

```bash
python scripts/01_prepare_data.py --top-dishes 100 --max-per-class 300
python scripts/02_train.py
python scripts/03_app.py
```

Etap 1 można powtarzać: już pobrane obrazy są pomijane. Cache obrazów w etapie 2 jest ważny tylko dla tej samej listy plików, a przy zmianie danych jest liczony od nowa.

### 8.3. Predykcja z linii poleceń

```bash
python scripts/03_app.py --image sciezka/do/zdjecia.jpg
```

Wynik jest wypisywany jako JSON (patrz [rozdział 11](#11-użycie-programistyczne-api)).

### 8.4. Trening na słabym sprzęcie (CPU)

```bash
python scripts/01_prepare_data.py --top-dishes 30 --max-per-class 200
python scripts/02_train.py --img-size 64 --width 16 --epochs 20
```

Alternatywą jest uruchomienie tych samych skryptów w usłudze z darmowym GPU (np. Google Colab lub Kaggle) i pobranie pliku `artifacts/food_recognizer.joblib`.

---

## 9. Parametry konfiguracyjne

### 9.1. `01_prepare_data.py`

| Parametr | Domyślnie | Znaczenie |
|---|---|---|
| `--top-dishes` | 100 | liczba najczęstszych dań (klas) |
| `--top-ingredients` | 150 | rozmiar słownika składników |
| `--min-per-class` | 30 | minimalna liczba przykładów, aby danie weszło do puli |
| `--max-per-class` | 400 | maksymalna liczba przykładów na danie |
| `--min-food-prob` | 0,5 | minimalne `food_prob` |
| `--camera-only` | wyłączone | tylko zdjęcia wykonane aparatem (wg prawdopodobieństw) |
| `--max-side` | 384 | maksymalny dłuższy bok zapisywanego obrazu (px) |
| `--workers` | 16 | liczba wątków pobierania |

### 9.2. `02_train.py`

| Parametr | Domyślnie | Znaczenie |
|---|---|---|
| `--img-size` | 128 | bok obrazu wejściowego (px). Mniej = szybciej, ale mniej szczegółów |
| `--width` | 32 | liczba filtrów w pierwszej warstwie (kolejne etapy: ×1, ×2, ×4, ×8) |
| `--epochs` | 30 | maksymalna liczba epok |
| `--batch-size` | 64 | liczba obrazów w jednej aktualizacji wag |
| `--lr` | 0,002 | szczytowy współczynnik uczenia |
| `--patience` | 6 | liczba epok bez poprawy walidacji przed zatrzymaniem |
| `--seed` | 42 | ziarno losowości (powtarzalność) |

Parametry ustawiane w konstruktorze `FoodNetEstimator` (plik `foodnet.py`), bez flag w linii poleceń:

| Parametr | Domyślnie | Znaczenie |
|---|---|---|
| `weight_decay` | 0,05 | zanik wag w AdamW (regularyzacja) |
| `dropout` | 0,3 | odsetek wyłączanych neuronów w warstwie wspólnej |
| `label_smoothing` | 0,1 | wygładzanie etykiet dania |
| `loss_weights` | (1, 5, 1) | wagi strat: danie, składniki, odżywianie |

### 9.3. `03_app.py`

| Parametr | Znaczenie |
|---|---|
| `--model` | ścieżka do pliku modelu (domyślnie `artifacts/food_recognizer.joblib`) |
| `--image` | ścieżka do zdjęcia, tryb CLI (bez uruchamiania interfejsu) |
| `--share` | publiczny link Gradio |

---

## 10. Wyniki i metryki

Po treningu `02_train.py` wypisuje metryki i zapisuje je w `artifacts/results/test_report.json`. Struktura pliku:

```json
{
  "dish":        { "accuracy": …, "balanced_accuracy": …, "macro_f1": …, "weighted_f1": …,
                   "top3_accuracy": …, "top5_accuracy": … },
  "ingredients": { "micro_f1": …, "macro_f1": …, "samples_f1": …, "jaccard_samples": …,
                   "hamming_loss": …, "mAP": … },
  "nutrition":   { "calories_kcal_mae": …, "calories_kcal_rmse": …, "calories_kcal_r2": …,
                   "calories_kcal_mae_baseline": …, "…": "(to samo dla protein_g, fat_g,
                   carbohydrate_g, total_weight_g)" },
  "params":      { … parametry estymatora … }
}
```

### 10.1. Definicje metryk

**Danie (klasyfikacja wieloklasowa)**

| Metryka | Opis |
|---|---|
| `accuracy` | odsetek poprawnie rozpoznanych dań |
| `balanced_accuracy` | średnia czułość po klasach (odporna na niezbalansowanie) |
| `macro_f1` / `weighted_f1` | F1 uśrednione po klasach (równo / ważone liczebnością) |
| `top3_accuracy`, `top5_accuracy` | czy prawdziwe danie jest wśród 3 (5) najbardziej prawdopodobnych |

**Składniki (klasyfikacja wieloetykietowa)**, nazewnictwo wg Gibaja i Ventura (2015), rozdz. 3.1.1:

| Metryka | Rodzaj | Opis |
|---|---|---|
| `micro_f1` | oparta na etykietach | F1 liczone ze zsumowanych TP/FP/FN wszystkich etykiet (dominują częste składniki) |
| `macro_f1` | oparta na etykietach | średnia F1 po etykietach (równa waga każdego składnika, czuła na rzadkie) |
| `samples_f1` | oparta na przykładach | F1 liczone dla każdego zdjęcia, a potem uśrednione |
| `jaccard_samples` | oparta na przykładach | \|Z∩Y\| / \|Z∪Y\| uśrednione po zdjęciach (w artykule nazwana *accuracy*) |
| `hamming_loss` | oparta na przykładach | odsetek błędnie zaklasyfikowanych par (zdjęcie, składnik), im mniej tym lepiej |
| `mAP` | oparta na etykietach | średnia po składnikach z *average precision* liczonego z prawdopodobieństw. Uwaga: to inna miara niż *average precision* z artykułu, który jest metryką rankingową liczoną dla każdego przykładu. W raporcie nazywaj ją „średnią po etykietach z average precision" |

Nie jest liczona dokładność 0/1 (*subset accuracy*, dopasowanie całego zbioru składników) ani miary rankingowe (*coverage*, *ranking loss*).

**Wartości odżywcze (regresja)**

| Metryka | Opis |
|---|---|
| `*_mae` | średni błąd bezwzględny (w jednostkach oryginalnych) |
| `*_rmse` | pierwiastek ze średniego błędu kwadratowego |
| `*_r2` | współczynnik determinacji (0 = tak dobrze jak średnia, 1 = idealnie, wartości ujemne = gorzej niż średnia) |
| `*_mae_baseline` | MAE modelu, który zawsze przewiduje średnią. Punkt odniesienia: model ma sens, jeśli `mae` jest wyraźnie mniejsze |

Metryki regresji liczone są tylko dla pól, w których wartość prawdziwa jest znana.

### 10.2. Wykresy

| Plik | Zawartość |
|---|---|
| `training_curves.png` | strata uczenia i walidacji oraz dokładność dania na walidacji w kolejnych epokach (pozwala ocenić przeuczenie) |
| `test_confusion_dishes.png` | znormalizowana macierz pomyłek dla 20 najczęstszych dań |
| `test_ingredient_f1.png` | F1 dla 30 najczęstszych składników |
| `test_nutrition_scatter.png` | wartość prawdziwa vs przewidywana dla każdego celu regresji (linia przerywana = idealna predykcja) |

---

## 11. Użycie programistyczne (API)

### 11.1. Predykcja

```python
from foodrec import config
from foodrec.pipeline import FoodRecognizer

system = FoodRecognizer.load(config.MODEL_PATH)
result = system.predict(["zdjecie.jpg"], top_k=3)[0]   # także obiekty PIL.Image

print(result)
```

Format wyniku:

```python
{
  "dish": [("pho", 0.41), ("noodle soup", 0.22), ("beef noodle", 0.09)],
  "ingredients": [("noodle", 0.93), ("beef", 0.71), ("green onion", 0.55)],
  "nutrition": {"calories_kcal": 452.3, "protein_g": 24.1, "fat_g": 14.8,
                "carbohydrate_g": 61.0, "total_weight_g": 410.0}
}
```

Nazwy dań i składników są po normalizacji (małe litery, liczba pojedyncza składników). Wartości odżywcze są **szacunkowe**.

### 11.2. Użycie estymatora bezpośrednio

```python
from foodrec.foodnet import FoodNetEstimator

est = FoodNetEstimator(width=32, epochs=30)          # parametry jak w sklearn
est.fit(X_train, y_train, eval_set=(X_val, y_val))   # y_* = słownik (patrz 6.1)

proba   = est.predict_proba(X_test)                  # (N, liczba_dań)
dishes  = est.predict(X_test)                        # indeksy dań
ing     = est.predict_ingredients(X_test)            # macierz 0/1 (N, liczba_składników)
nutr    = est.predict_nutrition(X_test)              # (N, 5), w jednostkach oryginalnych
```

Indeksy dań i składników odpowiadają kolejności klas z `LabelEncoder` i `MultiLabelBinarizer` (są zapisane w `FoodRecognizer.dish_names` i `FoodRecognizer.ingredient_names`).

---

## 12. Zgodność z wytycznymi przedmiotu

| Wytyczna | Realizacja |
|---|---|
| Kod w języku Python | cały projekt |
| Zgodność z API i narzędziami scikit-learn | `FoodNetEstimator` dziedziczy po `BaseEstimator` (metody `fit`, `predict`, `predict_proba`, `get_params`); `LabelEncoder`, `MultiLabelBinarizer`, `train_test_split` i wszystkie metryki pochodzą z scikit-learn |
| Wizualizacje w matplotlib | `evaluation.py` |
| Metody w osobnych modułach, unikanie ponownego definiowania istniejących elementów | logika w pakiecie `foodrec` (po jednym module na odpowiedzialność); gotowe elementy (metryki, podział danych, kodowanie etykiet, optymalizator, warstwy sieci) pochodzą z bibliotek |
| Kod eksperymentów w pojedynczych skryptach | `scripts/01…03` |
| Działający prototyp systemu | `03_app.py` (Gradio / CLI) |
| Model tworzony od zera | sieć inicjalizowana losowo, bez pretrenowanych wag |
| Raport LNCS/IEEE (max 10 stron) oraz deklaracja użycia narzędzi AI | poza zakresem tej dokumentacji. **Deklarację należy dodać do raportu** zgodnie z wytycznymi PWr |

Uwaga: sieć jest zaimplementowana w PyTorch i opakowana w estymator w stylu scikit-learn. Jeżeli prowadzący rozumie „model od zera" jako implementację samych algorytmów (np. w numpy), warto to uzgodnić.

---

## 13. Ograniczenia i znane problemy

**Status weryfikacji.** Składnia wszystkich plików została sprawdzona, a funkcje parsujące dane przetestowano na przykładowych wierszach zbioru. **Pełny trening sieci nie został uruchomiony** na rzeczywistych danych (w środowisku, w którym powstał kod, nie dało się zainstalować PyTorcha), więc pierwszy przebieg może ujawnić drobne błędy w `foodnet.py`.

**Ograniczenia metodyczne i danych:**

- **Niespójne nazwy dań.** `dish_name` to tekst swobodny, a bardzo podobne dania mają różne nazwy („Noodle Soup", „Noodle Bowl", „Noodles in broth"), podczas gdy zdjęcia są niemal nieodróżnialne. Dokładność dania jest więc z natury ograniczona. Składniki i top-3 mogą wypaść lepiej.
- **Szablonowe wartości odżywcze.** W danych powtarzają się zaokrąglone liczby (np. 600 kcal), co wskazuje na szacunki, a nie pomiary. Regresja będzie słaba, a odniesienie do `*_mae_baseline` pokaże, czy cokolwiek wnosi.
- **Uczenie od zera na ograniczonej liczbie zdjęć** daje wyraźnie niższą jakość niż wykorzystanie sieci pretrenowanych. To oczekiwane i należy to opisać w raporcie.
- **Niezbalansowanie klas i składników.** Ograniczone przez `--max-per-class` i wybór najczęstszych klas, ale niewyeliminowane. Dla składników `macro_f1` będzie wyraźnie niższe niż `micro_f1`.
- **Założenie niezależności składników** (Binary Relevance), patrz 2.3.
- **Globalny próg** dla wszystkich składników. Osobne progi dla każdego składnika mogłyby poprawić wyniki rzadkich etykiet.
- **Duplikaty obrazów.** Usuwane są tylko powtórzone adresy URL. To samo zdjęcie pod różnymi adresami może trafić jednocześnie do zbioru uczącego i testowego i zawyżyć wyniki.
- **Dostępność obrazów.** Linki mogą być nieaktywne (są pomijane z komunikatem o liczbie martwych linków). Liczba pozostałych przykładów może być mniejsza od założonej.
- **Wartości odżywcze są szacunkowe** i nie mogą służyć do celów dietetycznych ani medycznych.
- **Ograniczenie słownika.** System rozpoznaje tylko dania i składniki, które znalazły się w wybranym zakresie `--top-dishes` i `--top-ingredients`. Nieznane danie zostanie przypisane do najbardziej podobnej znanej klasy.
- **Bezpieczeństwo pliku modelu.** `food_recognizer.joblib` jest serializowany przez `joblib`/`pickle`. Nie wczytuj plików modelu z niezaufanych źródeł.

---

## 14. Rozwiązywanie problemów

| Objaw | Możliwa przyczyna i rozwiązanie |
|---|---|
| `Brak metadata.parquet` | nie uruchomiono etapu 1: wykonaj `01_prepare_data.py` |
| Bardzo mało pobranych obrazów | wiele martwych linków lub blokada sieci: sprawdź komunikat „Pobrano X / Y", zmniejsz `--workers`, sprawdź połączenie |
| `MemoryError` / brak RAM przy tworzeniu tablicy obrazów | zmniejsz `--img-size` (np. 64) lub `--max-per-class` i `--top-dishes` w etapie 1 |
| Trening bardzo wolny | brak GPU: zmniejsz `--img-size`, `--width`, `--epochs` lub użyj Colab/Kaggle |
| `AttributeError: … GradScaler` | zbyt stara wersja PyTorcha: `pip install -U "torch>=2.3"` |
| `TypeError: … flagging_mode` | zbyt stara wersja Gradio: `pip install -U "gradio>=5.0"` |
| Strata walidacyjna rośnie, a uczenia maleje | przeuczenie: zwiększ augmentację lub `dropout`/`weight_decay`, zmniejsz `--width`, dodaj dane |
| Bardzo niska dokładność dania | zbyt wiele klas lub zbyt mało przykładów na klasę: zmniejsz `--top-dishes`, zwiększ `--max-per-class`, sprawdź `training_curves.png` |
| Model przewiduje wciąż te same składniki | niezbalansowanie: sprawdź `test_ingredient_f1.png` i próg w `test_report.json` |
| Błąd wczytania `.joblib` | inna wersja Pythona lub bibliotek niż przy zapisie: użyj tych samych wersji |

---

## 15. Możliwe rozszerzenia

- **Zależności między składnikami:** *Classifier Chains* lub funkcja straty BP-MLL (Gibaja i Ventura, 2015, rozdz. 4.1.5 i 4.2.4).
- **Osobne progi dla składników** dobierane na walidacji.
- **Obsługa niezbalansowania:** ważona strata (np. `pos_weight` w binarnej entropii krzyżowej) lub funkcja *focal loss*.
- **Stratyfikacja wieloetykietowa** podziału danych (Sechidis i in., 2011).
- **Wykrywanie duplikatów obrazów** (skróty percepcyjne) przed podziałem.
- **Walidacja krzyżowa** i testy statystyczne (Friedman, Wilcoxon) przy porównywaniu wariantów modelu.
- **Użycie kolumn `food_type` i `cooking_method`** jako dodatkowych zadań lub cech.
- **Szacowanie porcji** z uwzględnieniem skali obiektów na zdjęciu.

---

## 16. Bibliografia
**Dane:** karta zbioru MM-Food-100K na Hugging Face (https://huggingface.co/datasets/Humanbased-AI/MM-Food-100K). Warunki licencyjne zbioru należy sprawdzić na tej karcie przed publikacją wyników.

---
