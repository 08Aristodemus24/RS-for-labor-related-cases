import React, {useEffect} from "react";
import GraphVisualizer from "../components/GraphVisualizer";
import RequestManager from "../utils/requestManager";
import {useGlobalState} from "../state";

const {orderBy} = require('natural-orderby');

function Demo() {
    const [availableItems, setAvailableItems] = useGlobalState('availableItems');
    const [selectedFeatureSet, setSelectedFeatureSet] = useGlobalState('selectedFeatureSet');
    const [data, setData] = useGlobalState('data');
    const [task, setTask] = useGlobalState('task');
    const [subgraph, setSubgraph] = useGlobalState('subgraph');
    const [dataset, setDataset] = useGlobalState('dataset');
    const [options, setOptions] = useGlobalState('options');

    useEffect(() => {
        let option = localStorage.getItem('options');

        if (option === null) {
            localStorage.setItem('options', JSON.stringify(options));
            option = options;
        } else {
            option = JSON.parse(option);
            setOptions(option);
        }

        RequestManager.get("/available")
            .then((res) => RequestManager.validateResponse(res))
            .then((data) => {
                const selectedDataset = option.default.dataset || Object.keys(data.datasets)[0];
                setDataset(selectedDataset);

                let selectedTask;
                if (option.default.task && data.datasets[selectedDataset].supported_tasks.includes(option.default.task)) {
                    selectedTask = option.default.task;
                } else {
                    selectedTask = data.datasets[selectedDataset].supported_tasks[0];
                }

                setTask(selectedTask);

                setAvailableItems({
                    tasks: data.tasks,
                    datasets: data.datasets
                });

                const featureSets = orderBy(Object.keys(data.datasets[selectedDataset].attributes));

                if (typeof (option.default.featureSet[selectedDataset]) !== "undefined" && Object.keys(data.datasets[selectedDataset].attributes).includes(options.default.featureSet[selectedDataset])) {
                    setSelectedFeatureSet(option.default.featureSet[selectedDataset]);
                } else {
                    setSelectedFeatureSet(featureSets.length > 0 ? featureSets[featureSets.length - 1] : "n/a");
                }

                const subgraphID = option.default.subgraph[selectedDataset] || 0;
                setSubgraph(subgraphID);
            })
            .catch((err) => {
                console.error(err);
            })
    }, []);

    useEffect(() => {
        if (subgraph > -1 && dataset && task) {
            RequestManager.get(`/subgraphs/${subgraph}?dataset=${dataset}&task=${task}`)
                .then((res) => RequestManager.validateResponse(res))
                .then((data) => {
                    setData({
                        links: data.graph.links,
                        nodes: data.graph.nodes,
                        duplicateMapping: data.duplicate_mapping,
                        mapping: data.mapping,
                        groupedPersonAttributes: data.grouped_person_attributes
                    });
                })
                .catch((err) => {
                    console.error(err);
                })
        }
    }, [subgraph, dataset, task]);

    return (
        <div>
            <GraphVisualizer/>
        </div>
    );
}

export default Demo;
