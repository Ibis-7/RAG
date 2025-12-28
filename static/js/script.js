// document.addEventListener('DOMContentLoaded', () => {
//     const chatWindow = document.getElementById('chat-window');
//     const userInput = document.getElementById('user-input');
//     const sendBtn = document.getElementById('send-btn');
//     const sourcesList = document.getElementById('sources-list');

//     const handleChat = async () => {
//         const query = userInput.value.trim();
//         if (!query) return;

//         // Add User Message
//         appendMessage(query, 'user');
//         userInput.value = '';

//         // Create empty Bot Message bubble
//         const botMsgDiv = document.createElement('div');
//         botMsgDiv.className = 'bot-msg';
//         botMsgDiv.innerText = "...";
//         chatWindow.appendChild(botMsgDiv);
//         chatWindow.scrollTop = chatWindow.scrollHeight;

//         try {
//             const response = await fetch('/ask', {
//                 method: 'POST',
//                 headers: { 'Content-Type': 'application/json' },
//                 body: JSON.stringify({ query: query })
//             });

//             const reader = response.body.getReader();
//             const decoder = new TextDecoder();
//             let accumulatedText = "";

//             while (true) {
//                 const { value, done } = await reader.read();
//                 if (done) break;

//                 const chunk = decoder.decode(value);
//                 const lines = chunk.split('\n');

//                 for (const line of lines) {
//                     if (line.startsWith('data: ')) {
//                         const rawData = line.replace('data: ', '').trim();
//                         if (rawData === '[DONE]') break;

//                         try {
//                             const data = JSON.parse(rawData);
                            
//                             if (data.type === 'token') {
//                                 accumulatedText += data.content;
//                                 botMsgDiv.innerText = accumulatedText; // Real-time update
//                             } 
//                             else if (data.type === 'sources') {
//                                 renderSources(data.content);
//                             }
//                         } catch (e) { console.error("Error parsing JSON", e); }
//                     }
//                 }
//                 chatWindow.scrollTop = chatWindow.scrollHeight;
//             }
//         } catch (err) {
//             botMsgDiv.innerText = "Error connecting to server.";
//             console.error(err);
//         }
//     };

//     function renderSources(sources) {
//         sourcesList.innerHTML = '<h2 class="font-bold mb-2">Sources:</h2>';
//         sources.forEach(s => {
//             const sDiv = document.createElement('div');
//             sDiv.className = 'source-card p-2 mb-2 border rounded bg-gray-50 text-xs';
//             // Cleans up file path for display
//             const fileName = s.source.split(/[\\/]/).pop();
//             sDiv.innerHTML = `<strong>${fileName}</strong><p>${s.preview}...</p>`;
//             sourcesList.appendChild(sDiv);
//         });
//     }

//     function appendMessage(text, type) {
//         const msgDiv = document.createElement('div');
//         msgDiv.className = type === 'user' ? 'user-msg ml-auto bg-blue-500 text-white p-2 rounded mb-2 w-fit' : 'bot-msg bg-gray-200 p-2 rounded mb-2 w-fit';
//         msgDiv.innerText = text;
//         chatWindow.appendChild(msgDiv);
//     }

//     sendBtn.addEventListener('click', handleChat);
//     userInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') handleChat(); });
// });

document.addEventListener('DOMContentLoaded', () => {
    const chatWindow = document.getElementById('chat-window');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const sourcesList = document.getElementById('sources-list');

    const handleChat = async () => {
        const query = userInput.value.trim();
        if (!query) return;

        // 1. Add User Message to UI
        appendMessage(query, 'user');
        userInput.value = '';

        // 2. Add temporary "Thinking..." bubble
        const botMsgDiv = document.createElement('div');
        botMsgDiv.className = 'bot-msg bg-gray-200 p-2 rounded mb-2 w-fit';
        botMsgDiv.innerText = "Thinking...";
        chatWindow.appendChild(botMsgDiv);
        chatWindow.scrollTop = chatWindow.scrollHeight;

        try {
            // 3. Send standard POST request
            const response = await fetch('/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: query })
            });

            const data = await response.json();

            if (data.status === 'success') {
                // 4. Replace "Thinking..." with the actual answer
                botMsgDiv.innerText = data.answer;
                
                // 5. Update the sources sidebar
                renderSources(data.sources);
            } else {
                botMsgDiv.innerText = "Error: " + (data.error || data.message);
            }
        } catch (err) {
            botMsgDiv.innerText = "Could not connect to the server.";
            console.error(err);
        }
        
        chatWindow.scrollTop = chatWindow.scrollHeight;
    };

    function renderSources(sources) {
        sourcesList.innerHTML = '<h2 class="font-bold mb-2">Sources:</h2>';
        if (!sources || sources.length === 0) {
            sourcesList.innerHTML += '<p class="text-xs text-slate-400">No sources found.</p>';
            return;
        }

        sources.forEach(s => {
            const sDiv = document.createElement('div');
            sDiv.className = 'source-card p-2 mb-2 border rounded bg-gray-50 text-xs';
            const fileName = s.source.split(/[\\/]/).pop();
            sDiv.innerHTML = `<strong>${fileName}</strong><p>${s.preview || ''}...</p>`;
            sourcesList.appendChild(sDiv);
        });
    }

    function appendMessage(text, type) {
        const msgDiv = document.createElement('div');
        // Matches your existing Tailwind styles
        msgDiv.className = type === 'user' 
            ? 'user-msg ml-auto bg-blue-500 text-white p-2 rounded mb-2 w-fit' 
            : 'bot-msg bg-gray-200 p-2 rounded mb-2 w-fit';
        msgDiv.innerText = text;
        chatWindow.appendChild(msgDiv);
    }

    sendBtn.addEventListener('click', handleChat);
    userInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') handleChat(); });
});