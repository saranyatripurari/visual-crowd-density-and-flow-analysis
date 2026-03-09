document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("uploadForm");
    const loader = document.getElementById("loader");

    if (form) {
        form.addEventListener("submit", function () {
            loader.style.display = "block";
        });
    }
});
