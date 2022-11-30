import React, {useEffect, useState} from "react";
import "./scss/PredictionComparer.scss";
import {useGlobalState} from "../state";
import RequestManager from "../utils/requestManager";

const queryString = require('query-string');

function PredictionComparer() {
    const [selectedFeatureSet] = useGlobalState('selectedFeatureSet');
    const [availableItems] = useGlobalState('availableItems');
    const [subgraph] = useGlobalState('subgraph');
    const [options] = useGlobalState('options');
    const [dataset] = useGlobalState('dataset');
    const [task] = useGlobalState('task');
    const [selectedNodes, setSelectedNodes] = useGlobalState('selectedNodes');
    const [highlightedNodes, setHighlightedNodes] = useGlobalState('highlightedNodes');
    const [predictions, setPredictions] = useState([]);
    const [rankedLists, setRankedLists] = useState([]);

    const fetchSimilarityScores = async () => {
        const models = availableItems.datasets[dataset].models[selectedFeatureSet][task];

        const promises = models.map((model) => {
            const node1 = selectedNodes[0];
            const node2 = selectedNodes[1];
            const query = queryString.stringify({
                model: model.unique_name,
                features: selectedFeatureSet,
                dataset: dataset,
                task: task
            });

            const url = `/subgraphs/${subgraph}/similarity/${node1}/${node2}?${query}`;
            return RequestManager.get(url);
        });

        const results = await Promise.all(promises);

        return results.map((result, i) => ({
            model: models[i].display_name,
            value: result.data.predictions[0].toFixed(6)
        }));
    };

    const fetchRankedList = async () => {
        const models = availableItems.datasets[dataset].models[selectedFeatureSet][task];
        const promises = models.map((model) => {
            const node1 = selectedNodes[0];
            const node2 = 0;
            const query = queryString.stringify({
                model: model.unique_name,
                features: selectedFeatureSet,
                dataset: dataset,
                task: task,
                all: "true"
            });

            const url = `/subgraphs/${subgraph}/similarity/${node1}/${node2}?${query}`;
            return RequestManager.get(url);
        });

        const results = await Promise.all(promises);

        return results.map((result, i) => {
            let preds = result.data.predictions.map((prediction, i) => {
                const nodeID = result.data.request.nodes[1][i];

                if (nodeID.toString() !== selectedNodes[0]) {
                    return {
                        value: prediction.toFixed(6),
                        nodeID: nodeID
                    }
                }
            });

            preds.sort(function compare(a, b) {
                if (a.value < b.value) {
                    return 1;
                }
                if (a.value > b.value) {
                    return -1;
                }
                return 0;
            });

            return {
                model: models[i].display_name,
                predictions: preds
            };
        });
    };

    const addSelectedNode = (nodeID) => {
        setSelectedNodes([...selectedNodes, nodeID.toString()]);
        removeHighlightedNode(nodeID);
    };

    const addHighlightedNode = (nodeID) => {
        if(!highlightedNodes.includes(nodeID.toString())){
            setHighlightedNodes([...highlightedNodes, nodeID.toString()]);
        }
    };

    const removeHighlightedNode = (nodeID) => {
        const newHighlightedNodes = highlightedNodes.filter(e => e !== nodeID.toString());
        setHighlightedNodes(newHighlightedNodes)
    };

    useEffect(() => {
        if (selectedNodes.length === 2) {
            setRankedLists([]);
            fetchSimilarityScores()
                .then(predictions => setPredictions(predictions))
        } else if (selectedNodes.length === 1 && options.additionalFeatures.showRankedList === true) {
            setPredictions([]);
            fetchRankedList()
                .then(rankedLists => setRankedLists(rankedLists))
        }
    }, [selectedFeatureSet, selectedNodes]);

    return (
        <div className="predictionComparerContainer" style={{height: selectedNodes.length === 1 ? "72vh" : undefined}}>
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
            {
                rankedLists.length > 0 ?
                    <div className="rankedListTitle">
                        Ranked list
                    </div>
                    : undefined
            }
            {
                rankedLists.map((rankedItem, i) =>
                    <div className="rankedListItem" key={`${rankedItem.model}${i}`}>
                        <b>{rankedItem.model}</b>
                        <ul>
                            {
                                rankedItem.predictions.map((prediction, j) => {
                                    if (prediction) {
                                        return <li key={j} onClick={() => addSelectedNode(prediction.nodeID)}
                                                   onMouseOver={() => addHighlightedNode(prediction.nodeID)}
                                                   onMouseLeave={() => removeHighlightedNode(prediction.nodeID)}>
                                        <span>
                                            <a className="nodeID">
                                                {
                                                    `${prediction.nodeID}: `
                                                }
                                            </a>
                                        </span>
                                            {
                                                prediction.value
                                            }
                                        </li>;
                                    }
                                })
                            }
                        </ul>
                    </div>
                )
            }
        </div>
    );
}

export default PredictionComparer;
