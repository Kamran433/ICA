document.addEventListener('DOMContentLoaded', function () {
    const fileInput = document.getElementById('file');
    const filePreviewInput = document.getElementById('file-preview');
    const statusContainer = document.getElementById('status-container');
    const previewContainer = document.getElementById('file-preview-container');
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');

    // Establish a Socket.IO connection
    const socket = io.connect('http://' + document.domain + ':' + location.port);

    // Listen for 'status' events and update the message container
    socket.on('status', function (message) {
        updateStatus(message);
    });

    // Listen for 'image_paths' events and update the image preview
    socket.on('image_paths', function (channelPaths) {
        updateImagePreview(channelPaths);
    });

    // Interactive File Input for upload form
    fileInput.addEventListener('change', function () {
        previewFile(fileInput);
    });

    // Interactive File Input for preview
    filePreviewInput.addEventListener('change', function () {
        previewFile(filePreviewInput);
    });

    function previewFile(inputElement) {
        const fileType = inputElement.value.split('.').pop().toLowerCase();

        // Clear previous previews
        previewContainer.innerHTML = '';

        if (fileType === 'png') {
            // Display uploaded file
            const uploadedFilePreview = document.createElement('img');
            uploadedFilePreview.src = URL.createObjectURL(inputElement.files[0]);
            uploadedFilePreview.alt = 'Uploaded File Preview';
            uploadedFilePreview.classList.add('uploaded-file-preview');
            previewContainer.appendChild(uploadedFilePreview);}
        else if (fileType === 'jpg') {
            // Display uploaded file
            const uploadedFilePreview = document.createElement('img');
            uploadedFilePreview.src = URL.createObjectURL(inputElement.files[0]);
            uploadedFilePreview.alt = 'Uploaded File Preview';
            uploadedFilePreview.classList.add('uploaded-file-preview');
            previewContainer.appendChild(uploadedFilePreview);}
        else if (fileType === 'jpeg') {
            // Display uploaded file
            const uploadedFilePreview = document.createElement('img');
            uploadedFilePreview.src = URL.createObjectURL(inputElement.files[0]);
            uploadedFilePreview.alt = 'Uploaded File Preview';
            uploadedFilePreview.classList.add('uploaded-file-preview');
            previewContainer.appendChild(uploadedFilePreview);
        } else if (fileType === 'wav') {
            // Display audio preview
            const audioPreview = document.createElement('audio');
            audioPreview.controls = true;
            audioPreview.src = URL.createObjectURL(inputElement.files[0]);
            audioPreview.alt = 'Audio Preview';
            previewContainer.appendChild(audioPreview);}
        else if (fileType === 'mp3') {
            // Display audio preview
            const audioPreview = document.createElement('audio');
            audioPreview.controls = true;
            audioPreview.src = URL.createObjectURL(inputElement.files[0]);
            audioPreview.alt = 'Audio Preview';
            previewContainer.appendChild(audioPreview);
        }
        else if (fileType === 'm4a') {
            // Display audio preview
            const audioPreview = document.createElement('audio');
            audioPreview.controls = true;
            audioPreview.src = URL.createObjectURL(inputElement.files[0]);
            audioPreview.alt = 'Audio Preview';
            previewContainer.appendChild(audioPreview);
        }
         else {
            // Unsupported file type
            const unsupportedMessage = document.createElement('p');
            unsupportedMessage.textContent = 'Unsupported file type';
            previewContainer.appendChild(unsupportedMessage);
        }

        // Emit event to server for image separation
        emitStatus('Processing file. Please wait...', 'info');
        progressBar.style.width = '0%'; // Reset progress bar
        socket.emit('start_processing', { fileType, separationType: 'audio', icaMethod: 'fastica' });
    }

    // Function to update the status messages
   // Function to update the status messages
    function updateStatus(message, type = 'info') {
        const statusMessage = document.createElement('p');
        statusMessage.textContent = message;
        statusMessage.classList.add(type);
        statusContainer.appendChild(statusMessage);
    }

    // Function to update the image preview
    function updateImagePreview(channelPaths) {
        // Clear previous previews
        previewContainer.innerHTML = '';

        // Display image previews
        channelPaths.forEach((channel, index) => {
            const imgPreview = document.createElement('img');
            imgPreview.src = `data:image/png;base64,${channel}`;
            imgPreview.alt = `Separated Channel ${index + 1}`;
            imgPreview.classList.add('file-preview-container');
            previewContainer.appendChild(imgPreview);
        });

        // Log the channelPaths to the console for debugging
        console.log('Channel Paths:', channelPaths);
    }
});
