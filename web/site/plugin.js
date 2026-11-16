// plugin.js — progressive-enhanced <dialog> lightbox for the screenshot
// thumbnails on plugin.html. Plain left-click opens the modal in place;
// modified clicks (ctrl/meta/shift/alt or middle-click) fall through to
// the underlying <a href="…png">.

"use strict";
(function () {
  var dlg = document.getElementById("lightbox");
  var fullImg = document.getElementById("lightbox-img");
  var cap = document.getElementById("lightbox-cap");
  var closeBtn = document.getElementById("lightbox-close");

  Array.prototype.forEach.call(
    document.querySelectorAll(".shot-link"),
    function (link) {
      link.addEventListener("click", function (ev) {
        if (
          ev.button !== 0 ||
          ev.metaKey ||
          ev.ctrlKey ||
          ev.shiftKey ||
          ev.altKey ||
          typeof dlg.showModal !== "function"
        ) {
          return;
        }
        ev.preventDefault();
        var thumb = link.querySelector("img");
        fullImg.src = link.getAttribute("href") || link.href;
        fullImg.alt = thumb ? thumb.alt : "";
        cap.textContent = link.dataset.caption || "";
        if (!dlg.open) {
          dlg.showModal();
        }
      });
    },
  );

  closeBtn.addEventListener("click", function () {
    dlg.close();
  });

  dlg.addEventListener("click", function (ev) {
    if (ev.target === dlg) {
      dlg.close();
    }
  });
})();
