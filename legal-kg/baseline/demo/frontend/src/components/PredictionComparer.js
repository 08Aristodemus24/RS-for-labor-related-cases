import React, {useEffect, useState} from "react";
import "./scss/PredictionComparer.scss";
import {useGlobalState} from "../state";
import RequestManager from "../utils/requestManager";

const queryString = require('query-string');

function PredictionComparer() {
    const [selectedFeatureSet] = useGlobalState('selectedFeatureSet');
    const [availableItems] = useGlobalState('availableItems');
    const [subgraph] = useGlobalState('subgraph');
    const [selectedNodes] = useGlobalState('selectedNodes');
    const [predictions, setPredictions] = useState([]);

    const fetchSimilarityScores = async () => {
        const models = availableItems.models[selectedFeatureSet];

        if (selectedNodes.length === 2) {
            const promises = models.map((model) => {
                const node1 = selectedNodes[0];
                const node2 = selectedNodes[1];
                const query = queryString.stringify({
                    model: model.unique_name,
                    features: selectedFeatureSet
                });

                const url = `/subgraphs/${subgraph}/similarity/${node1}/${node2}?${query}`;
                return RequestManager.get(url);
            });

            const results = await Promise.all(promises);

            return results.map((result, i) => ({
                model: models[i].display_name,
                value: result.data.prediction.toFixed(6)
            }));
        }
    };

    useEffect(() => {
        fetchSimilarityScores()
            .then(predictions => setPredictions(predictions))
    }, [selectedFeatureSet, selectedNodes]);

    return (
        <div className="predictionComparerContainer">
            <div className="box">
                {
                    predictions.map((prediction, i) => <div key={i} className="predictionContainer">
                        <b>
                            {
                                prediction.model
                            }
                        </b>
                        <br/>
                        <span>
                            {
                                prediction.value
                            }
                        </span>
                    </div>)
                }
            </div>
        </div>
    );
}

export default PredictionComparer;
