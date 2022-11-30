import * as React from "react";
import "./scss/Settings.scss";
import {useGlobalState} from "../state";
import {useState} from "react";

function Settings(props) {
    const [options, setOptions] = useGlobalState('options');
    const [availableItems] = useGlobalState('availableItems');
    const [datasetDefaultSelected, setDatasetDefaultSelected] = useState(options.default.dataset);

    const handleChange = (event, checkbox, number, object, secondOrderObject) => {
        let isUndefined = false;
        if (event.target.value === "") {
            isUndefined = true;
        }

        if (secondOrderObject) {
            return setOptions({
                ...options,
                [object]: {
                    ...options[object],
                    [secondOrderObject]: {
                        ...options[object][secondOrderObject],
                        [event.target.name]: checkbox ? event.target.checked : number ? parseInt(event.target.value) : isUndefined ? undefined : event.target.value
                    }
                }
            });
        }

        if (object) {
            return setOptions({
                ...options,
                [object]: {
                    ...options[object],
                    [event.target.name]: checkbox ? event.target.checked : number ? parseInt(event.target.value) : isUndefined ? undefined : event.target.value
                }
            });
        }

        setOptions({
            ...options,
            [event.target.name]: checkbox ? event.target.checked : number ? parseInt(event.target.value) : isUndefined ? undefined : event.target.value
        });
    };

    const close = () => {
        localStorage.setItem('options', JSON.stringify(options));
        props.close();
    };

    return (
        <div className="dataModalContainer">
            <div className="dataModalContentContainer">
                <div className="headerContainer">
                    <div className="titleContainer">
                        <h2 className="title">Options</h2>
                    </div>
                </div>
                <div className="dataModalContent">
                    <div>
                        <p>General</p>
                        <hr/>
                        <div className="switchContainer">
                            <label className="switch">
                                <input type="checkbox" name="showDebuggingMode"
                                       defaultChecked={options.showDebuggingMode}
                                       onChange={(e) => handleChange(e, true, false)}/>
                                <span className="slider round"/>
                            </label>
                            <span>Advanced Mode <i>(for debugging)</i></span>
                        </div>
                        <div className="switchContainer">
                            <label className="switch">
                                <input type="checkbox" name="sideMenuResizeable"
                                       defaultChecked={options.sideMenuResizeable}
                                       onChange={(e) => handleChange(e, true, false)}/>/>
                                <span className="slider round"/>
                            </label>
                            <span>Sidemenu resizeable</span>
                        </div>
                        <div className="switchContainer">
                            <label className="switch">
                                <input type="checkbox" name="showSideMenuOnStart"
                                       defaultChecked={options.showSideMenuOnStart}
                                       onChange={(e) => handleChange(e, true, false)}/>/>
                                <span className="slider round"/>
                            </label>
                            <span>Show sidemenu after initial reload</span>
                        </div>
                        <br/>
                        <p>
                            Defaults
                            <select defaultValue={datasetDefaultSelected}
                                    onChange={e => setDatasetDefaultSelected(e.target.value)}>
                                {
                                    Object.keys(availableItems.datasets).map((name, i) => (
                                        <option key={i}>
                                            {
                                                name
                                            }
                                        </option>
                                    ))
                                }
                            </select>
                        </p>
                        <hr/>
                        <div className="switchContainer larger">
                            <input className="inputValue" name="dataset"
                                   defaultValue={options.default.dataset}
                                   onChange={(e) => handleChange(e, false, false, "default")}
                                   type="text"/>
                            <span>Default Dataset</span>
                        </div>
                        <div className="switchContainer larger">
                            <input className="inputValue" name="task"
                                   defaultValue={options.default.task}
                                   onChange={(e) => handleChange(e, false, false, "default")}
                                   type="text"/>
                            <span>Default Task</span>
                        </div>
                        <div className="switchContainer">
                            <input className="inputValue" name={datasetDefaultSelected} min={0} max={5000}
                                   value={options.default.subgraph[datasetDefaultSelected] || ""}
                                   onChange={(e) => handleChange(e, false, true, "default", "subgraph")}
                                   type="number"/>
                            <span>Default Subgraph ID</span>
                        </div>
                        <div className="switchContainer">
                            <input className="inputValue" name={datasetDefaultSelected}
                                   value={options.default.featureSet[datasetDefaultSelected] || ""}
                                   onChange={(e) => handleChange(e, false, false, "default", "featureSet")}
                                   type="text"/>
                            <span>Default Feature Set Name</span>
                        </div>
                        <br/>
                        <p>Graph Visualization</p>
                        <hr/>
                        <div className="switchContainer">
                            <label className="switch">
                                <input type="checkbox" name="active"
                                       defaultChecked={options.duplicateNodeHighlighting.active}
                                       onChange={(e) => handleChange(e, true, false, "duplicateNodeHighlighting")}/>
                                <span className="slider round"/>
                            </label>
                            <span>Highlight nodes (ground truth)</span>
                        </div>
                        {
                            options.duplicateNodeHighlighting.active ?
                                <div className="switchContainer indented">
                                    <label className="switch">
                                        <input type="checkbox" name="changeLabelColor"
                                               defaultChecked={options.duplicateNodeHighlighting.changeLabelColor}
                                               onChange={(e) => handleChange(e, true, false, "duplicateNodeHighlighting")}/>
                                        <span className="slider round"/>
                                    </label>
                                    <span>Change node label color when selected</span>
                                </div> : undefined
                        }
                        <div className="switchContainer">
                            <label className="switch">
                                <input type="checkbox" name="active"
                                       defaultChecked={options.duplicateLinkHighlighting.active}
                                       onChange={(e) => handleChange(e, true, false, "duplicateLinkHighlighting")}/>
                                <span className="slider round"/>
                            </label>
                            <span>Highlight links between nodes (ground truth)</span>
                        </div>
                        {
                            options.duplicateLinkHighlighting.active ?
                                <div>
                                    <div className="switchContainer indented">
                                        <input className="inputValue" name="color"
                                               defaultValue={options.duplicateLinkHighlighting.color}
                                               onChange={(e) => handleChange(e, false, false, "duplicateLinkHighlighting")}
                                               type="text"/>
                                        <span>Highlight Color</span>
                                    </div>
                                    <div className="switchContainer indented">
                                        <input className="inputValue" name="connected"
                                               defaultValue={options.duplicateLinkHighlighting.strokeWidth.connected}
                                               onChange={(e) => handleChange(e, false, true, "duplicateLinkHighlighting",
                                                   "strokeWidth")} type="number" min={0}/>
                                        <span>Stroke Width - Already connected components</span>
                                    </div>
                                    <div className="switchContainer indented">
                                        <input className="inputValue" name="notConnected"
                                               defaultValue={options.duplicateLinkHighlighting.strokeWidth.notConnected}
                                               onChange={(e) => handleChange(e, false, true, "duplicateLinkHighlighting",
                                                   "strokeWidth")} type="number" min={0}/>
                                        <span>Stroke Width - Not connected components</span>
                                    </div>
                                </div> : undefined
                        }
                        <br/>
                        <p>Attribute Comparison</p>
                        <hr/>
                        <div className="switchContainer">
                            <label className="switch">
                                <input type="checkbox" name="showOnlyAvailableAttributes"
                                       defaultChecked={options.attributeComparison.showOnlyAvailableAttributes}
                                       onChange={(e) => handleChange(e, true, false, "attributeComparison")}/>
                                <span className="slider round"/>
                            </label>
                            <span>Show only available attributes</span>
                        </div>
                        <div className="switchContainer">
                            <label className="switch">
                                <input type="checkbox" name="showSubject"
                                       defaultChecked={options.attributeComparison.showSubject}
                                       onChange={(e) => handleChange(e, true, false, "attributeComparison")}/>
                                <span className="slider round"/>
                            </label>
                            <span>Include subject attribute</span>
                        </div>
                        <div className="switchContainer">
                            <label className="switch">
                                <input type="checkbox" name="showMatching"
                                       defaultChecked={options.attributeComparison.showMatching}
                                       onChange={(e) => handleChange(e, true, false, "attributeComparison")}/>
                                <span className="slider round"/>
                            </label>
                            <span>Highlight string matching between attributes</span>
                        </div>
                        {
                            options.attributeComparison.showMatching ?
                                <div>
                                    <div className="switchContainer indented">
                                        <input className="inputValue" name="color"
                                               defaultValue={options.attributeComparison.matchingStyle.color}
                                               onChange={(e) => handleChange(e, false, false, "attributeComparison", "matchingStyle")}
                                               type="text"/>
                                        <span>Highlight Color</span>
                                    </div>
                                </div> : undefined
                        }
                        <br/>
                        <p>Additional Features</p>
                        <hr/>
                        <div className="switchContainer">
                            <label className="switch">
                                <input type="checkbox" name="showRankedList"
                                       defaultChecked={options.additionalFeatures.showRankedList}
                                       onChange={(e) => handleChange(e, true, false, "additionalFeatures")}/>
                                <span className="slider round"/>
                            </label>
                            <span>Show ranked list when only one node is selected</span>
                        </div>
                    </div>
                </div>
                <div className="buttonContainer">
                    <button onClick={close} className="button">
                        Save and close
                    </button>
                </div>
            </div>
        </div>
    );
}


export default Settings;