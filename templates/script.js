document.addEventListener('DOMContentLoaded', function() {
    // Check current URL path
    const currentPath = window.location.pathname;
    
    // Handle refine page (/refine)
    if (currentPath === '/refine') {
        handleRefinePage();
    }
    // Handle result page (/result)
    else if (currentPath === '/result') {
        handleResultPage();
    }
});

function handleRefinePage() {
    // Refine page elements
    const form = document.querySelector('form');
    const refinementInput = document.getElementById('refinement');
    
    if (form && refinementInput) {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            const refinement = refinementInput.value;
            regenerateDesign(refinement);
        });
    }
}

function handleResultPage() {
    // Result page elements
    const regenerateBtn = document.getElementById('regenerateBtn');
    const resultImage = document.getElementById('resultImage');
    const currentPrompt = document.getElementById('currentPrompt');
    
    // Add generation text div if it doesn't exist
    if (!document.getElementById('generationText')) {
        const resultContainer = resultImage.parentElement;
        const generationTextDiv = document.createElement('div');
        generationTextDiv.className = 'mt-4 p-3 bg-gray-50 rounded text-sm text-gray-700';
        generationTextDiv.id = 'generationText';
        resultContainer.appendChild(generationTextDiv);
    }
    
    // Function to call API and update image
    function regenerateDesign(prompt) {
        // Show loading state
        resultImage.classList.add('opacity-50');
        regenerateBtn.disabled = true;
        regenerateBtn.classList.add('opacity-50');
        
        const loadingIndicator = document.createElement('div');
        loadingIndicator.className = 'text-center py-4';
        loadingIndicator.innerHTML = '<p>Generating design...</p>';
        resultImage.parentElement.appendChild(loadingIndicator);
        
        fetch('/api/regenerate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ prompt: prompt })
        })
        .then(response => response.json())
        .then(data => {
            resultImage.src = data.image_url;
            resultImage.classList.remove('opacity-50');
            
            // Update generation text if available
            const generationText = document.getElementById('generationText');
            if (data.generation_text && data.generation_text.trim() !== '') {
                generationText.textContent = data.generation_text;
                generationText.classList.remove('hidden');
            } else {
                generationText.classList.add('hidden');
            }
            
            // Remove loading indicator
            const loadingElement = document.querySelector('.text-center.py-4');
            if (loadingElement) {
                loadingElement.remove();
            }
            
            // Show error if any
            if (data.error) {
                alert('Error: ' + data.error);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Failed to regenerate design. Please try again.');
            resultImage.classList.remove('opacity-50');
            regenerateBtn.disabled = false;
            regenerateBtn.classList.remove('opacity-50');
            
            // Remove loading indicator
            const loadingElement = document.querySelector('.text-center.py-4');
            if (loadingElement) {
                loadingElement.remove();
            }
        });
    }
    
    // Add event listeners for result page
    if (regenerateBtn && currentPrompt) {
        regenerateBtn.addEventListener('click', function() {
            regenerateDesign(currentPrompt.textContent);
        });
    }
}