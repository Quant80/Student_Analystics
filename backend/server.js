const express = require('express');
const axios = require('axios');
const cors = require('cors');
require('dotenv').config();

const app = express();
app.use(express.json());
app.use(cors());

// Serve static files (e.g., images)
app.use(express.static('static'));

// Serve HTML templates
app.set('view engine', 'ejs');
app.set('views', './templates');

// Home route
app.get('/', (req, res) => {
    res.render('home');
});

// AI route
app.get('/ai', (req, res) => {
    res.render('ai');
});

// AI API endpoint
app.post('/api/ai', async (req, res) => {
    try {
        const userMessage = req.body.message;

        // Call OpenAI API
        const response = await callOpenAI(userMessage);

        res.json({ response });
    } catch (error) {
        console.error('AI error:', error);
        res.status(500).json({ error: 'AI service unavailable' });
    }
});

async function callOpenAI(prompt) {
    const apiKey = process.env.OPENAI_API_KEY;

    if (!apiKey) {
        throw new Error("OPENAI_API_KEY is not set in .env file");
    }

    try {
        const response = await axios.post(
            'https://api.openai.com/v1/chat/completions',
            {
                model: "gpt-3.5-turbo",
                messages: [
                    {
                        role: "system",
                        content: "You are DUTAi, a helpful AI assistant for Durban University of Technology students."
                    },
                    { role: "user", content: prompt }
                ],
                max_tokens: 500,
                temperature: 0.7
            },
            {
                headers: {
                    'Authorization': `Bearer ${apiKey}`,
                    'Content-Type': 'application/json'
                }
            }
        );

        return response.data.choices[0].message.content;
    } catch (error) {
        console.error('OpenAI API error:', error.response?.data || error.message);
        throw error;
    }
}

// Start server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});