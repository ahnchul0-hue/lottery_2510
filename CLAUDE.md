# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **documentation-only** repository for a Korean lottery prediction system that combines multiple machine learning approaches. The project uses a hybrid ML architecture integrating PyTorch Transformers and scikit-learn ensembles to predict lottery numbers based on historical data analysis.

**Note**: This repository currently contains only specifications, architecture documents, and requirements - no actual implementation code.

## Architecture Overview

### Hybrid ML System Design
The system is designed as a 3-layer hybrid architecture:

1. **PyTorch Transformer Layer**: Deep learning for pattern recognition in number sequences
   - Multi-head attention mechanism for capturing number relationships
   - Positional encoding for temporal sequence analysis
   - Target: ~512MB memory usage, <100ms inference

2. **scikit-learn Ensemble Layer**: Traditional ML for statistical feature optimization  
   - RandomForest + GradientBoosting + MLP combination
   - Statistical features: frequency, odd/even ratio, AC values, digit sums
   - Target: ~256MB memory usage, <50ms inference

3. **Advanced Ensemble Integration**: Combines both approaches
   - Weighted voting, stacking, and dynamic blending strategies
   - Machine-specific optimization (1호기, 2호기, 3호기)
   - Target: <200ms end-to-end response time

### Data Architecture
- **Primary Dataset**: res.json (140 rounds of historical lottery data, 1049-1186 draws)
- **Database**: SQLite with lottery_draws table for persistent storage
- **API Layer**: FastAPI with standardized JSON responses
- **Machine Differentiation**: 70%+ different predictions across lottery machines

## Key Technical Specifications

### Performance Requirements
- **Response Time**: <200ms (95th percentile)
- **Memory Usage**: <2GB total (PyTorch 512MB + sklearn 256MB + system overhead)
- **Throughput**: >1000 requests/minute
- **Availability**: 99.9% uptime target

### Machine-Specific Strategies
- **1호기 (Machine 1)**: Conservative strategy emphasizing frequent numbers and high AC values
- **2호기 (Machine 2)**: Balanced optimization focusing on perfect distribution and digit sum harmony  
- **3호기 (Machine 3)**: Creative approach preferring odd numbers and pattern diversity

### Data Validation Rules
- Winning numbers: 6 unique integers in range 1-45
- Statistical consistency: Verify odd/even ratios, high/low distributions, AC values, digit sums
- Cross-validation: Ensure res.json data matches calculated statistics

## Development Context

### Technology Stack (Planned)
- **Backend**: FastAPI 0.104+, Python 3.8+
- **ML Frameworks**: PyTorch 1.x, scikit-learn 1.3+
- **Database**: SQLite + SQLAlchemy
- **Async**: uvloop, asyncio
- **Deployment**: Docker + Docker Compose
- **Monitoring**: Prometheus + Grafana

### API Design
- Base URL: `http://localhost:8000`
- Main endpoints:
  - `POST /api/predictions/generate` - Generate lottery predictions
  - `POST /api/ensemble/predictions/` - Multi-model ensemble predictions
  - `GET /api/statistics/machine-analysis/{machine_type}` - Statistical analysis
  - `GET /api/monitoring/health` - System health check

### Security Considerations
- API key or JWT token authentication
- CORS configuration for web frontend
- Input validation and sanitization
- SQL injection prevention
- Rate limiting by user tier

## Important Notes for Development

### Memory Management
- Sequential model loading to avoid memory spikes
- GPU/CPU hybrid processing (PyTorch on GPU, sklearn on CPU)
- Batch processing optimization
- Automatic garbage collection and cache cleanup

### Performance Optimization
- Asynchronous parallel inference execution
- Model result caching strategies
- Compressed JSON serialization
- Connection pooling for database operations

### Data Quality Assurance
- Real-time validation pipeline for incoming data
- Automatic schema version migration
- Statistical anomaly detection
- Comprehensive data quality reporting

### Ethical Guidelines
This system is designed for:
- Statistical analysis and pattern recognition research
- Educational purposes in machine learning
- Academic study of lottery number distributions

**Important Disclaimers**:
- Results are not guaranteed
- No liability for investment losses
- Research and educational use only
- Predictions are based on historical patterns, not guaranteed outcomes

## Document References

- **PRD.md**: Complete product requirements and business objectives
- **HYBRID_ML_ARCHITECTURE.md**: Detailed technical architecture and implementation
- **API_DOCUMENTATION.md**: Complete API specification with examples
- **RES_JSON_SPECIFICATION.md**: Data format specifications and validation rules

## Development Workflow (When Implementing)

1. **Environment Setup**: Python 3.8+, install PyTorch, scikit-learn, FastAPI
2. **Data Preparation**: Load and validate res.json historical data  
3. **Model Training**: Train PyTorch transformer and sklearn ensembles separately
4. **API Development**: Implement FastAPI endpoints following API_DOCUMENTATION.md
5. **Integration Testing**: Verify hybrid model coordination and performance
6. **Deployment**: Docker containerization with monitoring setup

## Quality Standards

- **Code Quality**: Follow data validation rules from RES_JSON_SPECIFICATION.md
- **Performance**: Meet response time and memory usage targets
- **Security**: Implement authentication and input validation
- **Testing**: Comprehensive coverage of ML model accuracy and API endpoints
- **Documentation**: Maintain clear explanations of prediction methodologies