import React, {useEffect} from "react";
import GraphVisualizer from "../components/GraphVisualizer";
import RequestManager from "../utils/requestManager";
import {useGlobalState} from "../state";

const {orderBy} = require('natural-orderby');

function Demo() {
    const [availableItems, setAvailableItems] = useGlobalState('availableItems');
    const [selectedFeatureSet, setSelectedFeatureSet] = useGlobalState('selectedFeatureSet');
    const [data, setData] = useGlobalState('data');
    const [subgraph, setSubgraph] = useGlobalState('subgraph');
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
                setAvailableItems(data);

                const featureSets = orderBy(Object.keys(data.attributes));

                if (typeof (option.default.featureSet) !== "undefined") {
                    setSelectedFeatureSet(option.default.featureSet);
                } else {
                    setSelectedFeatureSet(featureSets.length > 0 ? featureSets[featureSets.length - 1] : "n/a");
                }

                const subgraphID = option.default.subgraph || 0;
                setSubgraph(subgraphID);

                return data.subgraphs[subgraphID];
            })
            .then((first) => RequestManager.get(`/subgraphs/${first}`))
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
    }, []);

    return (
        <div>
            <GraphVisualizer/>
        </div>
    );
}

export default Demo;
