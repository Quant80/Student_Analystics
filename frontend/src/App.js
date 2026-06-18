import React, { useState } from 'react';
import PredictionForm from './components/PredictionForm';

const App = () => {
  const [predictionType, setPredictionType] = useState(null);

  window.renderPredictionForm = (type) => {
    setPredictionType(type);
  };

  return (
    <div>
      {predictionType && <PredictionForm type={predictionType} />}
    </div>
  );
};

export default App;