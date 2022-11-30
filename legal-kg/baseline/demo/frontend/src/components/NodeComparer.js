import React from "react";
import "./scss/NodeComparer.scss";
import {useGlobalState} from '../state';

function NodeComparer() {
    const [data] = useGlobalState('data');
    const [selectedNodes] = useGlobalState('selectedNodes');
    const [selectedFeatureSet] = useGlobalState('selectedFeatureSet');
    const [options] = useGlobalState('options');
    const [availableItems] = useGlobalState('availableItems');

    const groupByAttribute = (node) => {
        const nodeData = data.groupedPersonAttributes[node];

        let attributeMap = {};
        let attributes = [];

        nodeData.forEach((tuple) => {
            attributeMap[tuple[0]] = [];
            attributes.push(tuple[0]);
        });

        nodeData.forEach((tuple) => {
            attributeMap[tuple[0]].push(tuple[1])
        });

        return {
            attributeMap,
            attributes
        };
    };

    const showAttributeComparison = () => {
        if (selectedNodes.length > 0) {
            let comparableData = [];
            let attributes = [];
            let commonAttributeHighlighting = {};

            selectedNodes.map((node) => {
                const groupedData = groupByAttribute(node);
                comparableData.push(groupedData.attributeMap);

                attributes = attributes.concat(groupedData.attributes);
            });

            if (options.attributeComparison.showOnlyAvailableAttributes) {
                attributes = availableItems.attributes[selectedFeatureSet].filter(value => -1 !== attributes.indexOf(value));
            } else {
                attributes = availableItems.attributes[selectedFeatureSet];
            }

            if (options.attributeComparison.showSubject) {
                attributes.push("subject");
            }

            attributes.map((attribute) => {
                commonAttributeHighlighting[attribute] = [];

                if (options.attributeComparison.showMatching === true) {
                    if (selectedNodes.length === 2) {
                        const leftAttributeValue = comparableData[0][attribute];
                        const rightAttributeValue = comparableData[1][attribute];
                        commonAttributeHighlighting[attribute] = [];

                        if (typeof (leftAttributeValue) !== "undefined" && typeof (rightAttributeValue) !== "undefined") {
                            const sameItems = leftAttributeValue.filter(value => -1 !== rightAttributeValue.indexOf(value));
                            commonAttributeHighlighting[attribute] = commonAttributeHighlighting[attribute].concat(sameItems);
                        }
                    }
                }
            });

            let uniqueAttributes = [...new Set(attributes)];
            uniqueAttributes.sort(function (a, b) {
                if (a === b) return 0;
                if (a === "subject") return 1;
                if (b === "subject") return -1;

                if (a < b)
                    return -1;
                if (a > b)
                    return 1;
                return 0;
            });

            return (
                <table>
                    <thead>
                    <tr>
                        <th/>
                        {
                            selectedNodes.map((nodeID, i) => (<th key={i}>
                                <strong>
                                    {
                                        Object.keys(data.mapping).find(key => data.mapping[key] === parseInt(nodeID)).replace(/_/g, " ")
                                    }
                                </strong>
                                {
                                    ` (${nodeID})`
                                }
                            </th>))
                        }
                    </tr>
                    </thead>
                    <tbody>
                    {
                        uniqueAttributes.map((attribute, i) => (
                            <tr key={i}>
                                <th>
                                    <div>
                                        <dfn title={attribute}><strong><i>{attribute}</i></strong></dfn>
                                    </div>
                                </th>
                                {
                                    selectedNodes.map((nodeID, j) => {
                                        if (comparableData[j].hasOwnProperty(attribute)) {
                                            return (
                                                <th key={j}>
                                                    {
                                                        comparableData[j][attribute].map((item, h) => (
                                                            <div key={h}
                                                                 style={commonAttributeHighlighting[attribute].includes(item) ? options.attributeComparison.matchingStyle : undefined}>
                                                                {
                                                                    item
                                                                }
                                                                <br/>
                                                            </div>
                                                        ))
                                                    }
                                                </th>
                                            );
                                        } else {
                                            return <th key={j}/>
                                        }
                                    })
                                }
                            </tr>
                        ))
                    }
                    </tbody>
                </table>
            );
        }
    };

    return (
        <div>
            <div className="comparisonWrapper">
                <div className="tableContainer">
                    {
                        showAttributeComparison()
                    }
                </div>
            </div>
        </div>
    );
}

export default NodeComparer;
