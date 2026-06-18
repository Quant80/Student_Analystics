import React, { useState } from 'react';
import axios from 'axios';

const PredictionForm = ({ type }) => {
  const [prediction, setPrediction] = useState(null);
  const [shapPlot, setShapPlot] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const formData = {
      department: e.target.department?.value || '',
      marital_status: e.target.marital_status.value,
      gender: e.target.gender.value,
      race: e.target.race.value,
      disability: e.target.disability.value,
      stud_type: e.target.stud_type.value,
      exam_month: e.target.exam_month.value,
      exam_type: e.target.exam_type.value,
      offer_type: e.target.offer_type.value,
      ften_status: e.target.ften_status.value,
      final_year: e.target.final_year?.value || '',
      residence: e.target.residence.value,
      bursary: e.target.bursary.value,
      age_group: e.target.age_group.value,
    };

    try {
      const endpoint = type === 'applied' ? '/predictions/applied-sciences' : '/predictions/engineering';
      const response = await axios.post(endpoint, formData);
      setPrediction(response.data.prediction);
      setShapPlot(response.data.shapPlot);
      setError(null);
    } catch (err) {
      setError('Error fetching prediction: ' + err.message);
      setPrediction(null);
      setShapPlot(null);
    }
  };

  return (
    <div>
      <h2>{type === 'applied' ? 'Applied Sciences' : 'Engineering'} Prediction</h2>
      <form onSubmit={handleSubmit}>
        {type === 'applied' && (
          <div>
            <label>Department:</label>
            <select name="department" required>
              <option value="MARITIME STUDIES">Maritime Studies</option>
              <option value="BIOTECHNOLOGY & FOOD SCIENCE">Biotechnology & Food Science</option>
              <option value="HORTICULTURE">Horticulture</option>
              <option value="CHEMISTRY">Chemistry</option>
              <option value="DEPT OF TEXTILE SC & APP TECH">Textile Science & App Tech</option>
              <option value="SPORT STUDIES">Sport Studies</option>
              <option value="FOOD & NUTRIT CONSUMER SCIENCE">Food & Nutrition Consumer Science</option>
              <option value="MATHEMATICS">Mathematics</option>
              <option value="FACULTY OFFICE-APPLIED SCIENCE">Faculty Office-Applied Science</option>
            </select>
          </div>
        )}
        <div>
          <label>Marital Status:</label>
          <select name="marital_status" required>
            {type === 'applied' ? (
              <>
                <option value="MARRIED">Married</option>
                <option value="SINGLE">Single</option>
                <option value="SI">SI</option>
              </>
            ) : (
              <>
                <option value="M">Married</option>
                <option value="S">Single</option>
              </>
            )}
          </select>
        </div>
        <div>
          <label>Gender:</label>
          <select name="gender" required>
            <option value="M">Male</option>
            <option value="F">Female</option>
          </select>
        </div>
        <div>
          <label>Race:</label>
          <select name="race" required>
            <option value="AFRICAN">African</option>
            <option value="INDIAN">Indian</option>
            <option value="WHITE">White</option>
            <option value="COLOURED">Coloured</option>
            <option value="NON SA">Non SA</option>
          </select>
        </div>
        <div>
          <label>Disability:</label>
          <select name="disability" required>
            <option value="N">No</option>
            <option value="Y">Yes</option>
          </select>
        </div>
        <div>
          <label>Student Type:</label>
          <select name="stud_type" required>
            <option value="NORMAL STUDENT">Normal Student</option>
            <option value="SADC STUDENT">SADC Student</option>
            <option value="INTERNATIONAL STUDENT">International Student</option>
            <option value="REFUGEE">Refugee</option>
            <option value="ASYLUM SEEKER">Asylum Seeker</option>
          </select>
        </div>
        <div>
          <label>Exam Month:</label>
          <input type="number" name="exam_month" required />
        </div>
        <div>
          <label>Exam Type:</label>
          <select name="exam_type" required>
            <option value="NORMAL EXAM">Normal Exam</option>
            <option value="RE-EXAM(SUPP-MIDYEAR/YEAR END)">Re-Exam</option>
            <option value="SPECIAL EXAM">Special Exam</option>
            <option value="EXPERIENTIAL TRAINING">Experiential Training</option>
          </select>
        </div>
        <div>
          <label>Offer Type:</label>
          <select name="offer_type" required>
            <option value="D1">D1</option>
            <option value="D3">D3</option>
            <option value="P1">P1</option>
            <option value="I1">I1</option>
            <option value="P3">P3</option>
            <option value="I3">I3</option>
            <option value="D2">D2</option>
            <option value="D6">D6</option>
            <option value="D8">D8</option>
            <option value="I7">I7</option>
            <option value="D7">D7</option>
            <option value="P7">P7</option>
          </select>
        </div>
        <div>
          <label>FTEN Status:</label>
          <select name="ften_status" required>
            <option value="N">N</option>
            <option value="F">F</option>
            <option value="T">T</option>
            <option value="E">E</option>
          </select>
        </div>
        {type === 'applied' && (
          <div>
            <label>Final Year:</label>
            <select name="final_year" required>
              <option value="Y">Yes</option>
              <option value="N">No</option>
            </select>
          </div>
        )}
        <div>
          <label>Residence:</label>
          <select name="residence" required>
            <option value="Y">Yes</option>
            <option value="N">No</option>
          </select>
        </div>
        <div>
          <label>Bursary:</label>
          <select name="bursary" required>
            <option value="Y">Yes</option>
            <option value="N">No</option>
          </select>
        </div>
        <div>
          <label>Age Group:</label>
          <select name="age_group" required>
            <option value="15-19">15-19</option>
            <option value="20-29">20-29</option>
            <option value="30-39">30-39</option>
            <option value="40-49">40-49</option>
            <option value=">50">>50</option>
          </select>
        </div>
        <button type="submit">Predict</button>
      </form>

      {error && <p style={{ color: 'red' }}>{error}</p>}

      {prediction && (
        <div>
          <h3>Prediction Result</h3>
          <p><strong>Result:</strong> {prediction.result}</p>
          <p><strong>Pass Probability:</strong> {(prediction.pass_probability * 100).toFixed(2)}%</p>
          <p><strong>Fail Probability:</strong> {(prediction.fail_probability * 100).toFixed(2)}%</p>
        </div>
      )}

      {shapPlot && (
        <div>
          <h3>SHAP Feature Importance</h3>
          <img src={shapPlot} alt="SHAP Feature Importance Plot" style={{ maxWidth: '100%' }} />
        </div>
      )}
    </div>
  );
};

export default PredictionForm;