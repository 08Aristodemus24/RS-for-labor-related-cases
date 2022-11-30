import React from "react";
import "./css/sideMenu.scss";
import NodeComparer from "../../components/NodeComparer";
import {useGlobalState} from "../../state";
import PredictionComparer from "../../components/PredictionComparer";

const {orderBy} = require('natural-orderby');

function SideMenu() {
    const [availableItems] = useGlobalState('availableItems');
    const [selectedFeatureSet, setSelectedFeatureSet] = useGlobalState('selectedFeatureSet');
    const [selectedNodes] = useGlobalState('selectedNodes');
    const [subgraph] = useGlobalState('subgraph');
    const [options] = useGlobalState('options');

    const changeFeatureSet = (featureSetKey) => {
        setSelectedFeatureSet(featureSetKey);
    };

    const renderFeatureSets = () => {
        const keys = orderBy(Object.keys(availableItems.attributes));

        return keys.map((key, i) => {
            const cssClass = selectedFeatureSet === key ? "selected" : "";
            return (
                <span key={i} onClick={() => changeFeatureSet(key)} className={cssClass}> {key} </span>);
        });
    };

    return (
        <div className="menuWrapper"
             style={{
                 width: `${options.sideMenuWidth}%`,
                 backgroundColor: `rgba(21, 41, 53, ${options.sideMenuOpacity})`
             }}>
            <div className="nameContainer">
                {
                    options.showDebuggingMode ? <b className="name">Subgraph {subgraph}</b> :
                        <b className="name">Demo Graph</b>
                }
                <br/>
                <i className="label">Corpus: {availableItems.graph}</i>
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
                                renderFeatureSets()
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
                selectedNodes.length === 2 ? <PredictionComparer/> : undefined
            }
            <NodeComparer/>
            <b className="bottomText">{'\u00A9'} IBM 2019</b>
        </div>
    );
}

export default SideMenu;
