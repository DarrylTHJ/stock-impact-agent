# Bursa Impact Explorer Chrome Extension

This development extension sends the current article URL to the locally running
Bursa Impact Explorer. The Python application, ChromaDB and Gemini key remain on
the local computer; the extension contains no API credentials.

## Install locally

1. Start the Streamlit application on its default port:

   ```powershell
   .\.venv\Scripts\python.exe -m streamlit run app.py
   ```

2. Open `chrome://extensions` in Chrome.
3. Enable **Developer mode**.
4. Select **Load unpacked**.
5. Choose this `chrome_extension` folder.
6. Pin **Bursa Impact Explorer** to the Chrome toolbar.

While viewing a public article, click the extension icon. Alternatively,
right-click the page or a news link and select **Analyse with Bursa Impact
Explorer**. A new tab opens, URL mode is selected, and analysis starts once.

If Streamlit is running on a different port, update `APP_URL` in
`service-worker.js`, then press the extension's reload button on
`chrome://extensions`.
