// Select the upload input and label elements
const fileInput = document.getElementById("resumeUpload");
const uploadLabel = document.querySelector(".upload-label span");

// Add event listener for file selection
fileInput.addEventListener("change", (event) => {
    if (fileInput.files.length) {
        const fileName = fileInput.files[0].name;
        uploadLabel.textContent = fileName; // Update label to show the file name
    } else {
        uploadLabel.textContent = "Click to Upload PDF"; // Reset if no file is selected
    }
});

document.getElementById("scoreButton").addEventListener("click", () => {
    const resultDiv = document.getElementById("result");
    const scoreSpan = document.getElementById("score");

    if (!fileInput.files.length) {
        alert("Please upload a PDF file!");
        return;
    }

    const file = fileInput.files[0];

    if (file.type !== "application/pdf") {
        alert("Only PDF files are allowed!");
        return;
    }

    const score = calculateResumeScore(file.name);

    scoreSpan.textContent = score;
    resultDiv.classList.remove("hidden");
});

function calculateResumeScore(fileName) {
    const randomScore = Math.floor(Math.random() * 30) + 70; // Score range: 70-100
    console.log(`Analyzing resume: ${fileName}`);
    return randomScore;
}
