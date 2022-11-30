import React from "react";
import "./scss/GraphVisualizer.scss";
import {useGlobalState} from '../state';

import {Graph} from "react-d3-graph";
import * as constants from "../utils/constants";

const queryString = require('query-string');

function GraphVisualizer() {
    const [data] = useGlobalState('data');
    const [config] = useGlobalState('config');
    const [options] = useGlobalState('options');
    const [selectedNodes, setSelectedNodes] = useGlobalState('selectedNodes');

    const onClickNode = (nodeId) => {
        if (selectedNodes.includes(nodeId)) {
            const newSelectedNodes = selectedNodes.filter((id) => {
                return id !== nodeId;
            });
            setSelectedNodes([...newSelectedNodes]);
        } else {
            if (selectedNodes.length < 2) {
                let newSelectedNodes = selectedNodes;
                newSelectedNodes.push(nodeId.toString());

                setSelectedNodes([...newSelectedNodes]);
            } else {
                setSelectedNodes([nodeId]);
            }
        }
    };

    const onRightClickNode = (event, nodeId) => {
        window.alert(`Right clicked node ${nodeId}`);
    };

    const onClickLink = (source, target) => {
        const newSelectedNodes = [source, target];
        setSelectedNodes([...newSelectedNodes]);
    };

    const count = (names) =>
        names.reduce((a, b) => ({
            ...a,
            [b]: (a[b] || 0) + 1
        }), {});

    const duplicates = (dict) => Object.keys(dict).filter((a) => dict[a] > 1);

    const rebuildColorMap = (colorMaps, involvedNodesDuplicates, node, i) => {
        if (typeof (colorMaps[node]) === "undefined") {
            if (involvedNodesDuplicates.includes(node)) {
                colorMaps[node] = [];
                colorMaps[node].push(constants.colors[i]);
            } else {
                colorMaps[node] = constants.colors[i];
            }
            return colorMaps;
        } else {
            colorMaps[node].push(constants.colors[i]);
        }
        return colorMaps;
    };

    const prepareVisualizationData = (data) => {
        const nodes = data.nodes;
        const links = data.links;
        const duplicateMapping = data.duplicateMapping;
        const mapping = data.mapping;

        const involvedNodes = [...duplicateMapping.map(mapping => (mapping[0].toString())),
            ...duplicateMapping.map(mapping => (mapping[1].toString()))];

        const involvedNodesCount = count(involvedNodes);
        const involvedNodesDuplicates = duplicates(involvedNodesCount);

        let colorMaps = {};
        duplicateMapping.map((mapping, i) => {
            const node1 = mapping[0].toString();
            const node2 = mapping[1].toString();

            colorMaps = rebuildColorMap(colorMaps, involvedNodesDuplicates, node1, i);
            colorMaps = rebuildColorMap(colorMaps, involvedNodesDuplicates, node2, i);
        });

        const newNodes = nodes.map(node => {
            let color = undefined;
            let svg = undefined;
            let svgSize = undefined;

            if (options.duplicateNodeHighlighting.active === true) {
                if (involvedNodes.includes(node.id.toString())) {
                    color = colorMaps[node.id.toString()];

                    if (typeof (color) === "object") {
                        svg = `/generate/svg?${queryString.stringify({colors: color})}`;
                        svgSize = 275;
                        color = undefined;
                    }
                }
            }

            if (selectedNodes.includes(node.id.toString())) {
                svg = require("./assets/round.png");
                svgSize = 325;
            }

            return Object.assign(node, {
                color: color,
                svg: svg,
                size: svgSize,
                name: Object.keys(mapping).find(key => mapping[key] === node.id)
            });
        });

        let newLinks = links;

        // highlighting links between duplicates (optional, default: false)
        if (options.duplicateLinkHighlighting.active === true) {
            let existingLinksMapping = [];

            const existingDuplicateLinks = links.map(link => {
                existingLinksMapping.push(`${link.source},${link.target}`);

                const isAlsoDuplicate = duplicateMapping.filter(mapping => {
                    if (mapping[0] === link.source && mapping[1] === link.target) {
                        return mapping;
                    }
                    if (mapping[1] === link.source && mapping[0] === link.target) {
                        return mapping;
                    }
                });

                if (isAlsoDuplicate.length > 0) {
                    return Object.assign(link, {
                        color: options.duplicateLinkHighlighting.color,
                        strokeWidth: options.duplicateLinkHighlighting.strokeWidth.connected
                    });
                }

                return link;
            });

            let notExistingDuplicateLinks = [];
            duplicateMapping.forEach(mapping => {
                if (!existingLinksMapping.includes(`${mapping[0]},${mapping[1]}`)) {
                    notExistingDuplicateLinks.push({
                        color: options.duplicateLinkHighlighting.color,
                        strokeWidth: options.duplicateLinkHighlighting.strokeWidth.notConnected,
                        source: mapping[0],
                        target: mapping[1]
                    });
                }
            });

            newLinks = [...existingDuplicateLinks, ...notExistingDuplicateLinks]
        }

        return {
            nodes: newNodes,
            links: newLinks
        }
    };

    return (
        <div>
            {
                data.nodes.length > 0 ? <Graph
                    id="graph-viz"
                    data={prepareVisualizationData(data)}
                    config={config}
                    onClickNode={onClickNode}
                    onRightClickNode={onRightClickNode}
                    onClickLink={onClickLink}
                /> : undefined
            }
        </div>
    );
}

export default GraphVisualizer;
