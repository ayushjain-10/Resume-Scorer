document.getElementById("resumeUpload").addEventListener("change", function () {
    const fileName = this.files[0]?.name || "Click to Upload PDF";
    document.querySelector(".upload-label span").textContent = fileName;
});

document.getElementById("scoreButton").addEventListener("click", async (event) => {
    event.preventDefault();

    const fileInput = document.getElementById("resumeUpload");
    const resultDiv = document.getElementById("result");
    const classificationResultDiv = document.getElementById("classification-result");
    const similarityResultDiv = document.getElementById("similarity-result");

    if (!fileInput.files.length) {
        alert("Please upload a PDF file!");
        return;
    }

    const file = fileInput.files[0];

    if (file.type !== "application/pdf" && file.type !== "image/png" && file.type !== "image/jpeg") {
        alert("Only PDF or image files are allowed!");
        return;
    }

    // Display loading state
    resultDiv.innerHTML = "Processing...";
    resultDiv.classList.remove("hidden");

    // Send the file to the backend API
    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("https://resumerater-final-falling-star-2923.fly.dev/classify/", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            throw new Error("Failed to classify the resume.");
        }

        const data = await response.json();

        if (data.error) {
            alert(data.error);
            return;
        }

        // Update the result display
        resultDiv.innerHTML = ""; // Clear previous content
        classificationResultDiv.innerHTML = `
            <h3>Job Category: ${data.category}</h3>
        `;
        similarityResultDiv.innerHTML = `
            <h3>Similarity Score: ${data.similarity_score}</h3>
            <h4>Top Similar Resumes:</h4>
            <ul>
                ${data.similar_resumes
                    .map(
                        (resume) => `<li><a href="${resume}" target="_blank">${resume}</a></li>`
                    )
                    .join("")}
            </ul>
        `;
        resultDiv.appendChild(classificationResultDiv);
        resultDiv.appendChild(similarityResultDiv);
    } catch (error) {
        console.error(error);
        alert("An error occurred while classifying the resume.");
    }
});
