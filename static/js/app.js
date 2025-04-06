document.addEventListener('DOMContentLoaded', function() {
    // Get elements - check if they exist first
    const editPromptBtn = document.getElementById('editPromptBtn');
    const promptEditor = document.getElementById('promptEditor');
    const regenerateBtn = document.getElementById('regenerateBtn');
    const applyPromptBtn = document.getElementById('applyPromptBtn');
    const editedPrompt = document.getElementById('editedPrompt');
    const resultImage = document.getElementById('resultImage');
    
    // Only proceed with this functionality if we're on a page with the result image
    if (resultImage) {
        // Add a div for generation text
        const resultContainer = resultImage.parentElement;
        const generationTextDiv = document.createElement('div');
        generationTextDiv.className = 'mt-4 p-3 bg-gray-50 rounded text-sm text-gray-700';
        generationTextDiv.id = 'generationText';
        resultContainer.appendChild(generationTextDiv);
        
        // Function to call API and update image
        function regenerateDesign(prompt) {
            // Show loading state
            resultImage.classList.add('opacity-50');
            const loadingIndicator = document.createElement('div');
            loadingIndicator.className = 'text-center py-4';
            loadingIndicator.innerHTML = '<p>Generating design...</p>';
            resultContainer.appendChild(loadingIndicator);
            
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
                editedPrompt.value = data.prompt;
                promptEditor.classList.add('hidden');
                
                // Update generation text if available
                const generationText = document.getElementById('generationText');
                if (data.generation_text && data.generation_text.trim() !== '') {
                    generationText.textContent = data.generation_text;
                    generationText.classList.remove('hidden');
                } else {
                    generationText.classList.add('hidden');
                }
                
                // Remove loading indicator
                resultContainer.removeChild(loadingIndicator);
                
                // Show error if any
                if (data.error) {
                    alert('Error: ' + data.error);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Failed to regenerate design. Please try again.');
                resultImage.classList.remove('opacity-50');
                
                // Remove loading indicator
                if (resultContainer.contains(loadingIndicator)) {
                    resultContainer.removeChild(loadingIndicator);
                }
            });
        }
        
        // Attach event listeners only if elements exist
        if (editPromptBtn) {
            editPromptBtn.addEventListener('click', function() {
                promptEditor.classList.toggle('hidden');
            });
        }
        
        if (applyPromptBtn && editedPrompt) {
            applyPromptBtn.addEventListener('click', function() {
                regenerateDesign(editedPrompt.value);
            });
        }
        
        if (regenerateBtn && editedPrompt) {
            regenerateBtn.addEventListener('click', function() {
                regenerateDesign(editedPrompt.value);
            });
        }
    }
});

function toggleNewSession(checkbox) {
    document.getElementById('new_session').value = checkbox.checked ? 'true' : 'false';
}