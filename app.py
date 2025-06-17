"""
Author: Cameron Murphy (Student ID: 1049678, GitHub: 0x1049678II)
Date: June 4th 2025

Main Entry point for the Indra chatbot - Go Travel! project.

This is the main Flask application that serves the Indra chatbot web interface.
It handles HTTP requests, manages async bot interactions through event loops, and 
provides a web interface for users to interact with the travel assistant.

The app bridges the gap between synchronous Flask and the async Indra bot by creating
new event loops for bot operations, ensuring non-blocking behavior while maintaining
Flask's request-response pattern.
"""

import asyncio
import time

from structlog import get_logger, configure, processors, stdlib
from flask import Flask, request, jsonify, render_template
from flask_caching import Cache
from flask_cors import CORS

from chatbot.exceptions import IndraException, RateLimitExceededException
from chatbot.indra import IndraBot
from config.settings import Config

# Configure structured logging
configure(
    processors=[
        stdlib.filter_by_level,
        stdlib.add_logger_name,
        stdlib.add_log_level,
        stdlib.PositionalArgumentsFormatter(),
        processors.TimeStamper(fmt="iso"),
        processors.StackInfoRenderer(),
        processors.format_exc_info,
        processors.JSONRenderer()
    ],
    wrapper_class=stdlib.BoundLogger,
    logger_factory=stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = get_logger()

app = Flask(__name__)
app.config.from_object(Config)

# Initialize extensions
cors = CORS(app)
cache = Cache(app)

# Initialize Indra chatbot
indra_bot = IndraBot()


@app.route('/')
def index():
    """Serve the main chat interface."""
    return render_template('chat.html', bot_name='Indra')

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "Indra Travel Bot"})

@app.route('/chat', methods=['POST'])
def chat():
    """Main chat endpoint with async support."""
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({
                "status": "error",
                "error": "Message is required"
            }), 400

        user_message = data['message'].strip()
        if not user_message:
            return jsonify({
                "status": "error",
                "error": "Message cannot be empty"
            }), 400

        # Log the incoming request
        logger.info("Chat request received", user_message=user_message)

        # Get response from Indra bot (async)
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def get_bot_response():
                async with indra_bot as bot:
                    return await bot.get_response_async(user_message)
            
            bot_response = loop.run_until_complete(get_bot_response())
            loop.close()
            
            logger.info("Chat response generated", 
                       user_message=user_message, 
                       bot_response=bot_response[:100] + "..." if len(bot_response) > 100 else bot_response)
            
            return jsonify({
                "status": "success",
                "response": bot_response,
                "timestamp": time.time()
            })

        except RateLimitExceededException as e:
            logger.warning("Rate limit exceeded", error=str(e))
            return jsonify({
                "status": "error",
                "error": "Rate limit exceeded. Please try again later."
            }), 429

        except IndraException as e:
            logger.error("Indra chatbot error", error=str(e))
            return jsonify({
                "status": "error",
                "error": "I'm having trouble processing your request. Please try again."
            }), 500

    except Exception as e:
        logger.error("Unexpected error in chat endpoint", error=str(e))
        return jsonify({
            "status": "error",
            "error": "An unexpected error occurred"
        }), 500

@app.route('/locations')
@cache.cached(timeout=3600)  # Cache for 1 hour
def get_supported_locations():
    """Get list of the supported England locations."""
    return jsonify({
        "locations": Config.VALID_LOCATIONS,
        "count": len(Config.VALID_LOCATIONS)
    })

@app.route('/weather/<location>')
@cache.cached(timeout=1800)  # Cache for 30 minutes
def get_weather(location: str):
    """Get weather forecast for a specific location."""
    try:
        if location not in Config.VALID_LOCATIONS:
            return jsonify({
                "status": "error",
                "error": f"Location '{location}' is not supported. Check /locations for valid options."
            }), 400

        # Get weather data through Indra bot
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def get_weather_data():
            async with indra_bot as bot:
                return await bot.get_weather_forecast(location)
        
        weather_data = loop.run_until_complete(get_weather_data())
        loop.close()
        
        return jsonify({
            "status": "success",
            "location": location,
            "forecast": weather_data
        })

    except Exception as e:
        logger.error("Weather endpoint error", location=location, error=str(e))
        return jsonify({
            "status": "error",
            "error": "Unable to fetch weather data"
        }), 500

@app.route('/activities/<location>')
@cache.cached(timeout=3600)  # Cache for 1 hour
def get_activities(location: str):
    """Get activity recommendations for a specific location."""
    try:
        if location not in Config.VALID_LOCATIONS:
            return jsonify({
                "status": "error",
                "error": f"Location '{location}' is not supported. Check /locations for valid options."
            }), 400

        # Get activity recommendations through Indra bot
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def get_activity_data():
            async with indra_bot as bot:
                return await bot.get_activity_recommendations(location)
        
        activities = loop.run_until_complete(get_activity_data())
        loop.close()
        
        return jsonify({
            "status": "success",
            "location": location,
            "activities": activities
        })

    except Exception as e:
        logger.error("Activities endpoint error", location=location, error=str(e))
        return jsonify({
            "status": "error",
            "error": "Unable to fetch activity recommendations"
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

async def initialize_bot():
    """Initialize and train the chatbot on startup."""
    try:
        logger.info("Initializing Indra chatbot...")
        await indra_bot.initialize()
        logger.info("Indra chatbot initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize chatbot", error=str(e))
        raise

if __name__ == '__main__':
    # Initialize the bot
    asyncio.run(initialize_bot())
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=Config.DEBUG,
        threaded=True
    )