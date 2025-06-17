"""
Author: Cameron Murphy (Student ID: 1049678, GitHub: 0x1049678II)
Date: June 6th 2025

English locations configuration and data for the Indra chatbot.

Location data for the 10 supported England destinations including
coordinates, attractions, weather patterns, and travel recommendations. This data
drives the location validation, weather services, and activity recommendations.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

class RegionType(Enum):
    """Types of regions in England."""
    CITY = "city"
    HISTORIC_SITE = "historic_site"
    NATURAL_AREA = "natural_area"
    COASTAL = "coastal"
    UNIVERSITY_TOWN = "university_town"
    METROPOLITAN = "metropolitan"

@dataclass
class LocationInfo:
    """Location information."""
    name: str
    region_type: RegionType
    coordinates: Dict[str, float]
    description: str
    key_attractions: List[str]
    best_seasons: List[str]
    typical_weather: Dict[str, str]
    recommended_duration: str
    travel_tips: List[str]

class EnglandLocations:
    """England locations configuration with travel information."""
    
    LOCATIONS_DATA = {
        'Cumbria': LocationInfo(
            name='Cumbria',
            region_type=RegionType.NATURAL_AREA,
            coordinates={'lat': 54.4609, 'lon': -3.0886},
            description='Home to the Lake District, England\'s premier national park with stunning lakes, fells, and literary heritage.',
            key_attractions=[
                'Lake Windermere',
                'Scafell Pike',
                'Beatrix Potter attractions',
                'Keswick town',
                'Dove Cottage (Wordsworth\'s home)',
                'Grasmere village'
            ],
            best_seasons=['Spring', 'Summer', 'Early Autumn'],
            typical_weather={
                'spring': 'Mild and wet, 8-15°C',
                'summer': 'Pleasant and changeable, 12-20°C',
                'autumn': 'Cool and rainy, 6-14°C',
                'winter': 'Cold and wet, 2-8°C'
            },
            recommended_duration='3-5 days',
            travel_tips=[
                'Pack waterproof clothing year-round',
                'Book accommodations early in summer',
                'Consider staying in Keswick or Windermere as base',
                'Check weather before hiking'
            ]
        ),
        
        'Corfe Castle': LocationInfo(
            name='Corfe Castle',
            region_type=RegionType.HISTORIC_SITE,
            coordinates={'lat': 50.6395, 'lon': -2.0566},
            description='Dramatic ruined castle in Dorset, offering spectacular views and rich Norman history.',
            key_attractions=[
                'Corfe Castle ruins',
                'Corfe Castle village',
                'Swanage Railway',
                'Dorset countryside walks',
                'Studland Beach (nearby)',
                'Poole Harbour views'
            ],
            best_seasons=['Spring', 'Summer', 'Early Autumn'],
            typical_weather={
                'spring': 'Mild coastal climate, 9-16°C',
                'summer': 'Warm and generally dry, 14-22°C',
                'autumn': 'Mild but windier, 8-16°C',
                'winter': 'Cool and wet, 4-10°C'
            },
            recommended_duration='1-2 days',
            travel_tips=[
                'Combine with Dorset coast exploration',
                'Take the Swanage Railway for scenic journey',
                'Visit early morning for fewer crowds',
                'Wear comfortable walking shoes'
            ]
        ),
        
        'The Cotswolds': LocationInfo(
            name='The Cotswolds',
            region_type=RegionType.NATURAL_AREA,
            coordinates={'lat': 51.8330, 'lon': -1.8433},
            description='Area of Outstanding Natural Beauty featuring honey-colored stone villages and rolling hills.',
            key_attractions=[
                'Chipping Campden',
                'Bourton-on-the-Water',
                'Stow-on-the-Wold',
                'Bibury village',
                'Broadway Tower',
                'Cotswold Way walking trail'
            ],
            best_seasons=['Spring', 'Summer', 'Early Autumn'],
            typical_weather={
                'spring': 'Mild and fresh, 8-16°C',
                'summer': 'Warm and pleasant, 13-23°C',
                'autumn': 'Crisp and colorful, 7-17°C',
                'winter': 'Cool and often frosty, 2-9°C'
            },
            recommended_duration='2-4 days',
            travel_tips=[
                'Plan village-hopping route in advance',
                'Book country pubs for lunch',
                'Avoid summer weekends for fewer crowds',
                'Consider guided walking tours'
            ]
        ),
        
        'Cambridge': LocationInfo(
            name='Cambridge',
            region_type=RegionType.UNIVERSITY_TOWN,
            coordinates={'lat': 52.2053, 'lon': 0.1218},
            description='Historic university city famous for its prestigious colleges, punting, and academic atmosphere.',
            key_attractions=[
                'King\'s College Chapel',
                'Trinity College',
                'Punting on River Cam',
                'Fitzwilliam Museum',
                'Cambridge University Botanic Garden',
                'The Backs (college gardens)'
            ],
            best_seasons=['Spring', 'Summer', 'Early Autumn'],
            typical_weather={
                'spring': 'Mild and variable, 7-15°C',
                'summer': 'Warm and generally dry, 12-22°C',
                'autumn': 'Cool and crisp, 6-16°C',
                'winter': 'Cold and damp, 2-8°C'
            },
            recommended_duration='1-2 days',
            travel_tips=[
                'Book punting tours in advance during summer',
                'Wear comfortable walking shoes for cobblestones',
                'Visit colleges during opening hours',
                'Try traditional afternoon tea'
            ]
        ),
        
        'Bristol': LocationInfo(
            name='Bristol',
            region_type=RegionType.METROPOLITAN,
            coordinates={'lat': 51.4545, 'lon': -2.5879},
            description='Vibrant port city known for street art, maritime history, and cultural diversity.',
            key_attractions=[
                'Clifton Suspension Bridge',
                'SS Great Britain',
                'Bristol Harbourside',
                'Banksy street art trail',
                'St. Nicholas Market',
                'Bristol Cathedral'
            ],
            best_seasons=['Spring', 'Summer', 'Early Autumn'],
            typical_weather={
                'spring': 'Mild and showery, 8-16°C',
                'summer': 'Warm and pleasant, 13-22°C',
                'autumn': 'Mild but wet, 7-17°C',
                'winter': 'Cool and damp, 3-9°C'
            },
            recommended_duration='2-3 days',
            travel_tips=[
                'Explore different neighborhoods (Clifton, Harbourside)',
                'Check for street art tours',
                'Visit markets for local food',
                'Consider Bristol harbour ferry'
            ]
        ),
        
        'Oxford': LocationInfo(
            name='Oxford',
            region_type=RegionType.UNIVERSITY_TOWN,
            coordinates={'lat': 51.7520, 'lon': -1.2577},
            description='The \'City of Dreaming Spires\' - prestigious university city with stunning architecture.',
            key_attractions=[
                'Christ Church College',
                'Bodleian Library',
                'Radcliffe Camera',
                'Ashmolean Museum',
                'Carfax Tower',
                'Oxford Castle & Prison'
            ],
            best_seasons=['Spring', 'Summer', 'Early Autumn'],
            typical_weather={
                'spring': 'Mild and fresh, 7-16°C',
                'summer': 'Warm and generally pleasant, 12-23°C',
                'autumn': 'Cool and crisp, 6-17°C',
                'winter': 'Cold and damp, 2-8°C'
            },
            recommended_duration='1-2 days',
            travel_tips=[
                'Book college tours in advance',
                'Climb Carfax Tower for city views',
                'Visit covered market for shopping',
                'Take walking tour to learn history'
            ]
        ),
        
        'Norwich': LocationInfo(
            name='Norwich',
            region_type=RegionType.CITY,
            coordinates={'lat': 52.6309, 'lon': 1.2974},
            description='Medieval city in Norfolk with stunning cathedral, castle, and rich cultural heritage.',
            key_attractions=[
                'Norwich Cathedral',
                'Norwich Castle',
                'Elm Hill (medieval street)',
                'Norwich Market',
                'Sainsbury Centre for Visual Arts',
                'The Lanes shopping area'
            ],
            best_seasons=['Spring', 'Summer', 'Early Autumn'],
            typical_weather={
                'spring': 'Mild and dry, 7-15°C',
                'summer': 'Warm and generally dry, 12-21°C',
                'autumn': 'Cool and crisp, 6-16°C',
                'winter': 'Cold and damp, 2-7°C'
            },
            recommended_duration='1-2 days',
            travel_tips=[
                'Explore medieval streets on foot',
                'Visit market for local produce',
                'Check cathedral service times',
                'Consider Norfolk Broads day trip'
            ]
        ),
        
        'Stonehenge': LocationInfo(
            name='Stonehenge',
            region_type=RegionType.HISTORIC_SITE,
            coordinates={'lat': 51.1789, 'lon': -1.8262},
            description='Prehistoric monument and UNESCO World Heritage Site, one of the world\'s most famous stone circles.',
            key_attractions=[
                'Stonehenge stone circle',
                'Visitor Centre exhibitions',
                'Neolithic houses reconstruction',
                'Salisbury Cathedral (nearby)',
                'Old Sarum (nearby)',
                'Salisbury city center'
            ],
            best_seasons=['Spring', 'Summer', 'Early Autumn'],
            typical_weather={
                'spring': 'Mild and windy, 8-16°C',
                'summer': 'Warm but can be windy, 13-22°C',
                'autumn': 'Cool and breezy, 7-16°C',
                'winter': 'Cold and exposed, 3-9°C'
            },
            recommended_duration='Half day to 1 day',
            travel_tips=[
                'Book timed entry tickets in advance',
                'Dress warmly - site is very exposed',
                'Use audio guide for full experience',
                'Combine with Salisbury visit'
            ]
        ),
        
        'Watergate Bay': LocationInfo(
            name='Watergate Bay',
            region_type=RegionType.COASTAL,
            coordinates={'lat': 50.4429, 'lon': -5.0553},
            description='Spectacular beach in Cornwall perfect for surfing, water sports, and coastal walks.',
            key_attractions=[
                'Watergate Bay Beach',
                'Surfing and water sports',
                'Coastal cliff walks',
                'Beach restaurants',
                'Newquay attractions (nearby)',
                'Cornwall coast path'
            ],
            best_seasons=['Spring', 'Summer', 'Early Autumn'],
            typical_weather={
                'spring': 'Mild and changeable, 9-15°C',
                'summer': 'Warm and pleasant, 14-20°C',
                'autumn': 'Mild but stormy, 8-16°C',
                'winter': 'Cool and wet, 5-11°C'
            },
            recommended_duration='1-3 days',
            travel_tips=[
                'Check tide times for beach activities',
                'Bring windproof clothing',
                'Book surf lessons in advance',
                'Try local seafood restaurants'
            ]
        ),
        
        'Birmingham': LocationInfo(
            name='Birmingham',
            region_type=RegionType.METROPOLITAN,
            coordinates={'lat': 52.4862, 'lon': -1.8904},
            description='England\'s second-largest city, known for its industrial heritage, diverse culture, and excellent food scene.',
            key_attractions=[
                'Birmingham Museum and Art Gallery',
                'Jewellery Quarter',
                'Cadbury World',
                'Birmingham Back to Backs',
                'Bullring shopping center',
                'Gas Street Basin canals'
            ],
            best_seasons=['Spring', 'Summer', 'Early Autumn'],
            typical_weather={
                'spring': 'Mild and showery, 7-15°C',
                'summer': 'Warm and pleasant, 12-21°C',
                'autumn': 'Cool and wet, 6-16°C',
                'winter': 'Cold and damp, 2-8°C'
            },
            recommended_duration='1-2 days',
            travel_tips=[
                'Explore different quarters (Jewellery, Chinese)',
                'Try famous Birmingham curry houses',
                'Use extensive canal network for walks',
                'Visit markets for local atmosphere'
            ]
        ),
    }
    
    @classmethod
    def get_location(cls, name: str) -> Optional[LocationInfo]:
        """Get location information by name."""
        return cls.LOCATIONS_DATA.get(name)
    
    @classmethod
    def get_all_locations(cls) -> Dict[str, LocationInfo]:
        """Get all location information."""
        return cls.LOCATIONS_DATA.copy()
    
    @classmethod
    def get_locations_by_region_type(cls, region_type: RegionType) -> List[LocationInfo]:
        """Get locations filtered by region type."""
        return [location for location in cls.LOCATIONS_DATA.values() 
                if location.region_type == region_type]
    
    @classmethod
    def get_location_names(cls) -> List[str]:
        """Get list of all location names."""
        return list(cls.LOCATIONS_DATA.keys())
    
    @classmethod
    def get_coordinates(cls, name: str) -> Optional[Dict[str, float]]:
        """Get coordinates for a location."""
        location = cls.get_location(name)
        return location.coordinates if location else None
    
    @classmethod
    def get_weather_appropriate_activities(cls, location_name: str, weather_condition: str) -> List[str]:
        """Get activities suitable for specific weather conditions."""
        location = cls.get_location(location_name)
        if not location:
            return []
        
        weather_condition = weather_condition.lower()
        
        # Indoor activities for poor weather
        indoor_activities = {
            'Cambridge': ['Visit colleges', 'Fitzwilliam Museum', 'Shopping', 'Cafes and pubs'],
            'Oxford': ['College tours', 'Bodleian Library', 'Ashmolean Museum', 'Covered Market'],
            'Bristol': ['SS Great Britain', 'Museums', 'St. Nicholas Market', 'Indoor shopping'],
            'Birmingham': ['Museums', 'Jewellery Quarter tour', 'Shopping centers', 'Restaurant exploration'],
            'Norwich': ['Cathedral', 'Castle', 'Museums', 'Indoor market', 'The Lanes shopping']
        }
        
        # Outdoor activities for good weather
        outdoor_activities = {
            'Cumbria': ['Hiking', 'Lake activities', 'Scenic drives', 'Village walks'],
            'The Cotswolds': ['Village walks', 'Countryside hiking', 'Garden visits', 'Scenic drives'],
            'Watergate Bay': ['Surfing', 'Beach walks', 'Coastal hiking', 'Water sports'],
            'Corfe Castle': ['Castle exploration', 'Countryside walks', 'Railway journey'],
            'Stonehenge': ['Site visit', 'Archaeological exploration', 'Countryside walks']
        }
        
        if weather_condition in ['rain', 'cloudy', 'overcast', 'cold']:
            return indoor_activities.get(location_name, [])
        else:
            return outdoor_activities.get(location_name, location.key_attractions[:4])
    
    @classmethod
    def get_seasonal_recommendations(cls, location_name: str, season: str) -> Dict[str, Any]:
        """Get seasonal recommendations for a location."""
        location = cls.get_location(location_name)
        if not location:
            return {}
        
        season = season.lower()
        
        recommendations = {
            'activities': cls.get_weather_appropriate_activities(location_name, 'good'),
            'weather_info': location.typical_weather.get(season, 'Weather data not available'),
            'clothing_suggestions': [],
            'special_events': []
        }
        
        # Season-specific clothing suggestions
        clothing_map = {
            'spring': ['Light jacket', 'Umbrella', 'Comfortable walking shoes'],
            'summer': ['Light clothing', 'Sun hat', 'Sunscreen', 'Light rain jacket'],
            'autumn': ['Warm jacket', 'Waterproof clothing', 'Layers'],
            'winter': ['Warm coat', 'Waterproof clothing', 'Warm accessories', 'Sturdy footwear']
        }
        
        recommendations['clothing_suggestions'] = clothing_map.get(season, [])
        
        return recommendations