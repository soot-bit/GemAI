document.getElementById("predictionForm").addEventListener("submit", async function (e) {
    e.preventDefault();

    const loader = document.getElementById("loader");
    const resultCard = document.getElementById("result");
    const predictButton = this.querySelector("button[type='submit']");

    // Disable button and show loader
    predictButton.disabled = true;
    loader.style.display = "block";
    resultCard.style.display = "none";

    // Helper to get element value or default
    const getValue = (id, isNumeric = false) => {
        const element = document.getElementById(id);
        if (!element) {
            console.error(`Element with id '${id}' not found.`);
            return isNumeric ? 0 : "";
        }
        const value = element.value || "";
        return isNumeric ? parseFloat(value) : value;
    };

    const data = {
        carat: getValue("carat", true),
        cut: getValue("cut"),
        color: getValue("color"),
        clarity: getValue("clarity"),
        depth: getValue("depth", true),
        table: getValue("table", true),
        price: getValue("price", true),
        x: getValue("x", true),
        y: getValue("y", true),
        z: getValue("z", true),
    };

    //  delay 
    await new Promise(resolve => setTimeout(resolve, 500));

    try {
        const response = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data),
        });

        if (!response.ok) {
            const errorData = await response.json();
            const errorMessage = errorData?.detail || errorData?.message || "Request failed";
            resultCard.innerHTML = `
                <div class="error-card">
                    <h2>⚠️ Error</h2>
                    <p>${errorMessage}</p>
                </div>
            `;
            resultCard.style.display = "block";
            return;
        }

        const result = await response.json();

        function formatCurrency(value, currency) {
            return value.toLocaleString(undefined, {
                style: "currency",
                currency: currency,
                minimumFractionDigits: 2,
                maximumFractionDigits: 2,
            });
        }

        resultCard.innerHTML = `
            <h2>💎 Predicted Price</h2>
            <p><strong>USD:</strong> ${formatCurrency(result.usd, "USD")}</p>
            <p><strong>BWP:</strong> P ${result.bwp.toLocaleString(undefined, {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2,
            })}</p>
        `;
        resultCard.style.display = "block";

    } catch (err) {
        resultCard.innerHTML = `
            <div class="error-card">
                <h2>⚠️ An Unexpected Error Occurred</h2>
                <p>${err.message}</p>
                <p><strong>Suggestion:</strong> Please check your network connection or contact support.</p>
            </div>
        `;
        resultCard.style.display = "block";
    } finally {
        // Re-enable button and hide loader
        predictButton.disabled = false;
        loader.style.display = "none";
    }
});

// --- Image Modal Functionality ---
const mainImageWrapper = document.getElementById("mainImageWrapper");
const mainDiamondImage = mainImageWrapper.querySelector(".main-diamond-image");
const imageModal = document.getElementById("imageModal");
const modalImage = document.getElementById("modalImage");
const closeModalBtn = document.getElementById("closeModalBtn");

if (mainImageWrapper && mainDiamondImage && imageModal && modalImage && closeModalBtn) {
    mainImageWrapper.addEventListener("click", () => {
        modalImage.src = mainDiamondImage.src;
        imageModal.style.display = "block";
    });

    closeModalBtn.addEventListener("click", () => {
        imageModal.style.display = "none";
    });

    imageModal.addEventListener("click", (event) => {
        if (event.target === imageModal) {
            imageModal.style.display = "none";
        }
    });
}
