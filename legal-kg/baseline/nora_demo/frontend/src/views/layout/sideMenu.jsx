import React, {useState} from "react";
import "./css/sideMenu.scss";
import NodeComparer from "../../components/NodeComparer";
import {useGlobalState} from "../../state";
import PredictionComparer from "../../components/PredictionComparer";

const {orderBy} = require('natural-orderby');

function SideMenu() {
    const [availableItems] = useGlobalState('availableItems');
    const [selectedFeatureSet, setSelectedFeatureSet] = useGlobalState('selectedFeatureSet');
    const [selectedNodes, setSelectedNodes] = useGlobalState('selectedNodes');
    const [dataset, setDataset] = useGlobalState('dataset');
    const [subgraph, setSubgraph] = useGlobalState('subgraph');
    const [options] = useGlobalState('options');
    const [task, setTask] = useGlobalState('task');

    const changeFeatureSet = (featureSetKey) => {
        setSelectedFeatureSet(featureSetKey);
    };

    const changeTask = (taskName) => {
        if (task !== taskName) {
            let featureSet = Object.keys(availableItems.datasets[dataset].attributes)[0];
            if (options.default.featureSet[taskName] && Object.keys(availableItems.datasets[dataset].attributes).includes(options.default.featureSet[taskName])) {
                featureSet = options.default.featureSet[taskName];
            }

            setSelectedNodes([]);
            setTask(taskName);
            setSelectedFeatureSet(featureSet);
            setSubgraph(options.default.subgraph[dataset] || availableItems.datasets[dataset].subgraphs[0]);
        }
    };

    const changeDataset = (datasetName) => {
        if (dataset !== datasetName) {
            setSelectedNodes([]);

            const supportedTasks = availableItems.datasets[datasetName].supported_tasks;
            let newTask = task;
            if (!supportedTasks.includes(task)) {
                if (supportedTasks.length !== 0) {
                    newTask = supportedTasks[0];
                    setTask(newTask);
                }
            }

            setSelectedFeatureSet(options.default.featureSet[datasetName] || Object.keys(availableItems.datasets[datasetName].attributes)[0]);
            setDataset(datasetName);
            setSubgraph(options.default.subgraph[datasetName] || availableItems.datasets[datasetName].subgraphs[0]);
        }
    };

    const renderFeatureSets = () => {
        const keys = orderBy(Object.keys(availableItems.datasets[dataset].attributes));

        return keys.map((key, i) => {
            const cssClass = selectedFeatureSet === key ? "selected" : "";
            return (
                <span key={i} onClick={() => changeFeatureSet(key)} className={cssClass}> {key} </span>);
        });
    };

    const renderComparer = () => {
        if (selectedNodes.length === 0) {
            return;
        }
        if (selectedNodes.length === 1 && options.additionalFeatures.showRankedList === false) {
            return <NodeComparer/>;
        }
        if (selectedNodes.length === 2) {
            return (
                <div>
                    <PredictionComparer/>
                    <NodeComparer/>
                </div>
            );
        }

        return <PredictionComparer/>;
    };

    const renderDatasetType = () => {
        return (
            <div className="taskSelectorContainer">
                <i className="label">Corpus: <b>{dataset}</b></i>
                <div className="options">
                    {
                        Object.keys(availableItems.datasets).map((name, i) => (
                            <div key={i} className={dataset === name ? "selected" : undefined}
                                 onClick={() => changeDataset(name)}>
                                {
                                    name
                                }
                            </div>
                        ))
                    }
                </div>
            </div>
        );
    };

    const renderTaskType = () => {
        return (
            <div className="taskSelectorContainer">
                <b className="name">{availableItems.tasks[task] ? availableItems.tasks[task].short_name : undefined}</b>
                <div className="options">
                    {
                        Object.keys(availableItems.tasks).map((name, i) => {
                                if (availableItems.datasets[dataset].supported_tasks.includes(name)) {
                                    return (
                                        <div key={i} className={task === name ? "selected" : undefined}
                                             onClick={() => changeTask(name)}>
                                            {
                                                availableItems.tasks[name].display_name
                                            }
                                        </div>
                                    );
                                }
                            }
                        )
                    }
                </div>
            </div>
        );
    };

    return (
        <div className="menuWrapper"
             style={{
                 width: `${options.sideMenuWidth}%`,
                 backgroundColor: `rgba(21, 41, 53, ${options.sideMenuOpacity})`
             }}>
            <div className="nameContainer">
                {
                    renderDatasetType()
                }
                {
                    renderTaskType()
                }
            </div>
            <div className="selectorContainer">
                <div className="headerLabel">
                    <span>
                        Features
                    </span>
                </div>
                {
                    options.showDebuggingMode ?
                        <div className="expandContainer">
                            {
                                availableItems.datasets[dataset] ? renderFeatureSets() : undefined
                            }
                        </div> : undefined
                }
                <div className="modelSelectorContainer">
                    <span>
                        {
                            selectedFeatureSet
                        }
                    </span>
                </div>
            </div>
            {
                renderComparer()
            }
            <b className="bottomText">{'\u00A9'} IBM 2019</b>
        </div>
    );
}

export default SideMenu;
