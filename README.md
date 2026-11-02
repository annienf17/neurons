# EgoExplained — Neural Journey

Interaktywny prototyp 3D: HTML + CSS + vanilla JavaScript + Three.js (bez React i bez procesu build).

## Uruchomienie

1. Rozpakuj ZIP lub wejdź do folderu z `index.html`, `styles.css`, `main.js`.
2. W terminalu, będąc w tym folderze, uruchom:
   ```bash
   python3 -m http.server 8000
   ```
3. Otwórz **http://localhost:8000** w Chrome, Safari lub Firefox.

**Wymagany internet** przy pierwszym uruchomieniu: biblioteka Three.js jest ładowana z jsDelivr CDN (wersja 0.187.0), a fonty z Google Fonts. Nie potrzeba instalować pakietów npm.

## Sterowanie

- **Scroll / gładzik:** przelot w głąb sceny 3D, pięć rozdziałów.
- **Ruch kursora:** lekki parallax kamery + kolorowe smugi jak na wodzie.
- **Kliknięcie w scenę:** dwa rozszerzające się świetlne kręgi.
- **01–05 po prawej:** przejdź do wybranego rozdziału.
- **MOTION ON/OFF:** zatrzymaj ruch dodatkowy (przyjazne dla osób wrażliwych na animacje).

## Jak zmieniać wygląd

- `main.js` → `CONFIG`: gęstość sieci, długość podróży, bloom, render pixel ratio.
- `main.js` → `basePalette`: 6 kolorów neuronów; procedura `buildAuroras()` odpowiada za kolorowe mgły.
- `main.js` → `laneX/laneY`: droga kamery i zakręty, `animate()` — szybkość płynnego dochodzenia do pozycji (`cameraDamping`).
- `styles.css` → `:root`: typografia, hot-pink brand, rozmiary, mobile/RWD.
- `index.html` → `.chapter`: tytuły i treści kolejnych etapów.

## Architektura

- `Three.js` renderuje rzeczywiste trójwymiarowe neurony, dendryty, synaptyczne połączenia, punkty GPU, pierścienie i shaderowe zorze.
- `EffectComposer + UnrealBloomPass` nadają połączeniom świetlisty glow.
- `window.scrollY` mapuje trasę kamery z interpolacją czasową; bez manipulacji natywnym scrollem.
- osobny przezroczysty `<canvas id="ink">` generuje wodniste smugi kursora.
- wersja mobile obniża liczbę neuronów, drobin i rozdzielczość GPU.
- `prefers-reduced-motion` i przełącznik ograniczają animacje dodatkowe.

## Ważne

To niezależne DEMO, a nie zmiana istniejącej strony EgoExplained. Nie potrzebuje backendu ani dostępu do obecnego projektu. Na produkcji warto przetestować GPU/temperaturę i optymalizować na urządzeniach docelowych. Scena jest metaforą sieci neuronalnej, nie rekonstrukcją anatomiczną mózgu.
