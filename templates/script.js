    document.addEventListener('DOMContentLoaded', function() {
        const editPromptBtn = document.getElementById('editPromptBtn');
        const promptEditor = document.getElementById('promptEditor');
        const regenerateBtn = document.getElementById('regenerateBtn');
        const applyPromptBtn = document.getElementById('applyPromptBtn');
        const editedPrompt = document.getElementById('editedPrompt');
        const resultImage = document.getElementById('resultImage');
        
        // Add a div for generation text
        const resultContainer = resultImage.parentElement;
        const generationTextDiv = document.createElement('div');
        generationTextDiv.className = 'mt-4 p-3 bg-gray-50 rounded text-sm text-gray-700';
        generationTextDiv.id = 'generationText';
        resultContainer.appendChild(generationTextDiv);
        
        // Show/hide prompt editor
        editPromptBtn.addEventListener('click', function() {
            promptEditor.classList.toggle('hidden');
        });
        
        // Apply edited prompt and regenerate
        applyPromptBtn.addEventListener('click', function() {
            regenerateDesign(editedPrompt.value);
        });
        
        // Regenerate with current prompt
        regenerateBtn.addEventListener('click', function() {
            regenerateDesign(editedPrompt.value);
        });
        
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
    });

    // Add to static/js/app.js
function submitUploadForm() {
    const uploadForm = document.getElementById('upload-form');
    const progressContainer = document.getElementById('progress-container');
    const progressBar = document.getElementById('progress-bar');
    const statusText = document.getElementById('status-text');
    
    // Show progress container
    progressContainer.classList.remove('hidden');
    
    // Get form data
    const formData = new FormData(uploadForm);
    
    // Submit form via fetch
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            statusText.textContent = 'Error: ' + data.error;
            return;
        }
        
        // Start polling for status
        const jobId = data.job_id;
        pollJobStatus(jobId);
    })
    .catch(error => {
        console.error('Error:', error);
        statusText.textContent = 'Error uploading files';
    });
    
    // Prevent form submission
    return false;
}

function pollJobStatus(jobId) {
    const progressBar = document.getElementById('progress-bar');
    const statusText = document.getElementById('status-text');
    
    // Poll every 2 seconds
    const pollInterval = setInterval(() => {
        fetch(`/status/${jobId}`)
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    clearInterval(pollInterval);
                    statusText.textContent = 'Error: ' + data.error;
                    return;
                }
                
                // Update progress bar and status
                progressBar.style.width = `${data.progress}%`;
                
                // Update status text based on current operation
                switch (data.status) {
                    case 'uploading':
                        statusText.textContent = 'Uploading images to cloud storage...';
                        break;
                    case 'analyzing':
                        statusText.textContent = 'Analyzing fabric patterns with AI...';
                        break;
                    case 'generating_prompt':
                        statusText.textContent = 'Creating design prompt based on analysis...';
                        break;
                    case 'complete':
                        statusText.textContent = 'Processing complete!';
                        clearInterval(pollInterval);
                        // Redirect to next page
                        if (data.redirect) {
                            window.location.href = data.redirect;
                        }
                        break;
                    case 'error':
                        statusText.textContent = 'Error: ' + data.error;
                        clearInterval(pollInterval);
                        break;
                    default:
                        statusText.textContent = 'Processing...';
                }
            })
            .catch(error => {
                console.error('Error:', error);
                statusText.textContent = 'Error checking status';
                clearInterval(pollInterval);
            });
    }, 2000);
}