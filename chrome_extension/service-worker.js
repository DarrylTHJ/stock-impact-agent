const APP_URL = "http://localhost:8501/";

function openExplorer(articleUrl) {
  if (!articleUrl || !/^https?:\/\//i.test(articleUrl)) return;
  const destination = `${APP_URL}?article_url=${encodeURIComponent(articleUrl)}`;
  chrome.tabs.create({ url: destination });
}

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "analyse-with-bursa-impact-explorer",
    title: "Analyse with Bursa Impact Explorer",
    contexts: ["page", "link"]
  });
});

chrome.action.onClicked.addListener((tab) => {
  openExplorer(tab.url);
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId !== "analyse-with-bursa-impact-explorer") return;
  openExplorer(info.linkUrl || tab?.url);
});
