document.addEventListener("DOMContentLoaded", function() {
    const uploadToggle = document.getElementById("upload-toggle");
    const uploadForm = document.getElementById("upload-form");
    if (uploadToggle) {
        uploadToggle.addEventListener("click", function() {
            uploadForm.classList.toggle("hidden");
            uploadToggle.textContent = uploadForm.classList.contains("hidden")
                ? "📤 Upload & Analyze"
                : "❌ Hide Upload Form";
        });
    }
});
