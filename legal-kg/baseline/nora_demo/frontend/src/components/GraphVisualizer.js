import React from "react";
import "./scss/GraphVisualizer.scss";
import {useGlobalState} from '../state';

import {Graph} from "react-d3-graph";
import * as constants from "../utils/constants";

const queryString = require('query-string');

const electRepsFilterNames =
    [
        "Jerzy Buzek",
        "Guy Verhofstadt",
        "Jeremy Corbyn",
        "Garnett Genuis",
        "Bill Morneau",
        "Mairead McGuinness",
        "Chris Warkentin",
        "Antonio Tajani",
        "Jody Wilson-Raybould",
        "Markus Ferber",
        "Andrew Scheer",
        "Lisa Raitt",
        "József Szájer",
        "Tony Clement",
        "Diane James",
        "Molly Scott Cato",
        "Joyce Murray",
        "Liam Fox",
        "Carolyn Bennett",
        "Simon Marcil",
        "Nathan Cullen",
        "Philippe Lamberts",
        "Seb Dance",
        "Milan Zver",
        "Kay Swinburne",
        "Bill Etheridge",
        "Angelika Niebler",
        "Catherine McKenna",
        "Nigel Farage",
        "Jean Lambert",
        "Maxime Bernier",
        "Marc Garneau",
        "Paul Manly",
        "Peter Mandelson",
        "Elizabeth May",
        "Paul Nuttall",
        "Caroline Lucas",
        "Peter Julian",
        "Theresa May",
        "Brad Trost",
        "Janice Atkinson",
        "Monika Hohlmeier",
        "Marco Rubio",
        "Darshan Kang",
        "Jagmeet Singh",
        "Chrystia Freeland",
        "Justin Trudeau",
        "Larry Maguire",
        "John Bercow",
        "Viviane Reding",
        "Helga Stevens",
        "Steven Woolfe",
        "Nathan Gill",
        "Ted Cruz",
        "Patrick O'Flynn"
    ];

function GraphVisualizer() {
    const [data] = useGlobalState('data');
    const [config] = useGlobalState('config');
    const [task] = useGlobalState('task');
    const [dataset] = useGlobalState('dataset');
    const [subgraph] = useGlobalState('subgraph');
    const [options] = useGlobalState('options');
    const [selectedNodes, setSelectedNodes] = useGlobalState('selectedNodes');
    const [highlightedNodes] = useGlobalState('highlightedNodes');

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
        let nodes = data.nodes;
        let links = data.links;
        const duplicateMapping = data.duplicateMapping;
        const mapping = data.mapping;

        // filter nodes in elect. reps. dataset / subgraph 0 - experimental code
        if (dataset === "elective representative" && subgraph === 0) {
            const nodeIDs = electRepsFilterNames.map(name => {
                if (Object.keys(mapping).includes(name)) {
                    return mapping[name];
                }
            });

            if (typeof (nodeIDs[0]) !== "undefined") {
                nodes = nodes.filter((node) => nodeIDs.includes(node.id));
                links = links.filter((link) => nodeIDs.includes(link.source) && nodeIDs.includes(link.target))
            }
        }

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
            let fontColor = undefined;

            if (options.duplicateNodeHighlighting.active === true && task === "NORP") {
                if (involvedNodes.includes(node.id.toString())) {
                    color = colorMaps[node.id.toString()];

                    if (typeof (color) === "object") {
                        svg = `/generate/svg?${queryString.stringify({colors: color})}`;
                        svgSize = 275;
                    }
                }
            }

            if (selectedNodes.includes(node.id.toString()) || highlightedNodes.includes(node.id.toString())) {
                const highlightColor = color ? color : [config.node.color];
                svg = `/generate/svg?${queryString.stringify({colors: highlightColor, checked: true})}`;
                svgSize = 325;

                if (options.duplicateNodeHighlighting.changeLabelColor === true) {
                    fontColor = highlightColor;
                }
            }

            return Object.assign(node, {
                color: color,
                fontColor: fontColor,
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
                        type: "CURVE_SMOOTH",
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
