# Aplikacja `CoachStats`

## Wstęp

CoachStats to innowacyjne narzędzie służące do analizowania statystyk Expected Goals (XG) w piłce nożnej. Jest przeznaczony głównie dla trenerów oraz piłkarzy w celach analitycznych czy kibiców w celach rozrywkowych.

## Funkcje

-   **Analiza XG:** Analiza XG dla danej sytuacji meczowej.
-   **Personalizowana makieta:** Możliwość tworzenia własnej makiety odpowiadającej sytuacji meczowej.

## Uruchomienie

Aby uruchomić aktualną wersję aplikacji, wykonaj następujące kroki:

1.  **Uruchomienie serwera:** Aby uruchomić serwer należy wejść w [link](https://sg-server-ne4h.onrender.com/) oraz zaczekać do pojawienia się komunikatu o błędzie 404. Aplikacja korzysta z darmowej wersji obsługi serwera, stąd sposób jego uruchomienia nie jest standardowy.

2.  **Uruchomienie aplikacji webowej** Po wykonaiu kroku 1 można wejść na właściwą stronę [aplikacji](https://statgoals.onrender.com/).

## Dane

-   <https://statsbomb.com/what-we-do/soccer-data/>

## Szczegóły

Działanie aplikacji zostało opisane w podręczniku użytkownika znajdującym w repozytorium. Poza tym zachęcam do przejrzenia analizy danych na podstawie których stworzono model [klik-klik](https://github.com/LukaszChrostowski/CoachStats/blob/main/notebooks/data_analysis.ipynb), a także sam proces jego uczenia [klik-klik](https://github.com/LukaszChrostowski/CoachStats/blob/main/notebooks/xgboost.ipynb). Przetworzone dane znajdują się w [klik-klik](https://github.com/LukaszChrostowski/CoachStats/blob/main/notebooks/data_to_ml.csv).
